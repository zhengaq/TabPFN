def predict(  # noqa: C901, PLR0912
        self,
        X: XType,
        *,
        # TODO: support "ei", "pi"
        output_type: Literal[
            "mean",
            "median",
            "mode",
            "quantiles",
            "full",
            "main",
        ] = "mean",
        quantiles: list[float] | None = None,
    ) -> np.ndarray | list[np.ndarray] | MainOutputDict | FullOutputDict:
        """Predict the target variable.

        Args:
            X: The input data.
            output_type:
                Determines the type of output to return.

                - If `"mean"`, we return the mean over the predicted distribution.
                - If `"median"`, we return the median over the predicted distribution.
                - If `"mode"`, we return the mode over the predicted distribution.
                - If `"quantiles"`, we return the quantiles of the predicted
                    distribution. The parameter `output_quantiles` determines which
                    quantiles are returned.
                - If `"main"`, we return the all output types above in a dict.
                - If `"full"`, we return the full output of the model, including the
                  logits and the criterion, and all the output types from "main".

            quantiles:
                The quantiles to return if `output="quantiles"`.

                By default, the `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
                quantiles are returned. The predictions per quantile match
                the input order.

        Returns:
            The predicted target variable or a list of predictions per quantile.
        """
        check_is_fitted(self)

        X = validate_X_predict(X, self)
        X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)
        X = _process_text_na_dataframe(X, ord_encoder=self.preprocessor_)  # type: ignore

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


        outputs = None
        borders = None

        transformed_logits = [
            translate_probs_across_borders(
                logits,
                frm=torch.as_tensor(borders_t, device=self.device_),
                to=self.bardist_.borders.to(self.device_),
            )
            for logits, borders_t in zip(outputs, borders)
        ]
        stacked_logits = torch.stack(transformed_logits, dim=0)
        if self.average_before_softmax:
            logits = stacked_logits.log().mean(dim=0).softmax(dim=-1)
        else:
            logits = stacked_logits.mean(dim=0)

        # Post-process the logits
        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()
        logits = logits.cpu()

        # Determine and return intended output type
        logit_to_output = partial(
            _logits_to_output,
            logits=logits,
            criterion=self.renormalized_criterion_,
            quantiles=quantiles,
        )
        if output_type in ["full", "main"]:
            # Create a dictionary of outputs with proper typing via TypedDict
            # Get individual outputs with proper typing
            mean_out = typing.cast(np.ndarray, logit_to_output(output_type="mean"))
            median_out = typing.cast(np.ndarray, logit_to_output(output_type="median"))
            mode_out = typing.cast(np.ndarray, logit_to_output(output_type="mode"))
            quantiles_out = typing.cast(
                list[np.ndarray],
                logit_to_output(output_type="quantiles"),
            )

            # Create our typed dictionary
            main_outputs = MainOutputDict(
                mean=mean_out,
                median=median_out,
                mode=mode_out,
                quantiles=quantiles_out,
            )

            if output_type == "full":
                # Return full output with criterion and logits
                return FullOutputDict(
                    **main_outputs,
                    criterion=self.renormalized_criterion_,
                    logits=logits,
                )

            return main_outputs

        return logit_to_output(output_type=output_type)