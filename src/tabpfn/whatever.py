
        X = validate_X_predict(X, self)
        X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)
        X = _process_text_na_dataframe(X, ord_encoder=self.preprocessor_)  # type: ignore

        # --- Standard Prediction Logic ---
        # (This part is largely the same as the original code)
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            assert all(
                (0 <= q <= 1) and (isinstance(q, float)) for q in quantiles
            ), "All quantiles must be between 0 and 1 and floats."
        if output_type not in self._USABLE_OUTPUT_TYPES:
            raise ValueError(f"Invalid output type: {output_type}")

        std_borders = self.bardist_.borders.cpu().numpy()
        outputs: list[torch.Tensor] = []
        borders: list[np.ndarray] = []

        for output, config in self.executor_.iter_outputs(
            X,
            device=self.device_,
            autocast=self.use_autocast_,
        ):
            assert isinstance(config, RegressorEnsembleConfig)

            if self.softmax_temperature != 1:
                output = output.float() / self.softmax_temperature  # noqa: PLW2901

            borders_t: np.ndarray
            logit_cancel_mask: np.ndarray | None
            descending_borders: bool

            # TODO(eddiebergman): Maybe this could be parallelized or done in fit
            # but I somehow doubt it takes much time to be worth it.
            # One reason to make it worth it is if you want fast predictions, i.e.
            # don't re-do this each time.
            # However it gets a bit more difficult as you need to line up the
            # outputs from `iter_outputs` above (which may be in arbitrary order),
            # along with the specific config the output belongs to. This is because
            # the transformation done to the borders for a given output is dependant
            # upon the target_transform of the config.
            if config.target_transform is None:
                borders_t = std_borders.copy()
                logit_cancel_mask = None
                descending_borders = False
            else:
                logit_cancel_mask, descending_borders, borders_t = (
                    _transform_borders_one(
                        std_borders,
                        target_transform=config.target_transform,
                        repair_nan_borders_after_transform=self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                    )
                )
                if descending_borders:
                    borders_t = borders_t.flip(-1)  # type: ignore

            borders.append(borders_t)

            if logit_cancel_mask is not None:
                output = output.clone()  # noqa: PLW2901
                output[..., logit_cancel_mask] = float("-inf")

            outputs.append(output)  # type: ignore

        # --- Translate probs, average, get final logits (same as before) ---
        transformed_probs = [ # Use probs here not logits for translation
            translate_probs_across_borders(
                torch.softmax(logits, dim=-1), # Convert logits to probs first
                frm=torch.as_tensor(borders_t, device=self.device_),
                to=self.bardist_.borders.to(self.device_),
            )
            for logits, borders_t in zip(outputs, borders)
        ]
        stacked_probs = torch.stack(transformed_probs, dim=0) # [Ensemble, N_samples, N_classes]

        if self.average_before_softmax:
            # Need to average the *original* logits before softmax
            original_logits = torch.stack(outputs, dim=0) # [Ensemble, N_samples, N_classes]
            avg_logits = original_logits.mean(dim=0)
            # Now translate these averaged logits
            final_probs = translate_probs_across_borders(
                 torch.softmax(avg_logits, dim=-1), # Probs from averaged logits
                 frm=torch.as_tensor(np.mean(borders, axis=0), device=self.device_), # Use average border? Or need per-sample? Simpler: Use standardized target borders.
                 to=self.bardist_.borders.to(self.device_), # Standardized target borders
             )
        else:
            # Average probabilities after translation
            final_probs = stacked_probs.mean(dim=0) # [N_samples, N_classes]