import warnings
import torch
from gpytorch import settings
# from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.models.gp import GP

from .student_t_likelihood import StudentTLikelihood
from .multivariate_student_t import MultivariateStudentT
from .prediction_strategy_tp import StudentTPredictionStrategy


class ExactTP(GP):
    """
    The base class for any Student-t process latent function to be used in conjunction
    with exact inference.

    :param torch.Tensor train_inputs: (size n x d) The training features X.
    :param torch.Tensor train_targets: (size n) The training targets y.
    :param StudentTLikelihood likelihood: The Student's t-likelihood that defines
        the observational distribution.

    The forward function should describe how to compute the prior latent distribution
    on a given input. Typically, this will involve a mean and kernel function.
    The result must be a MultivariateNormal.

    Calling this model will return the posterior of the latent Student-t process when conditioned
    on the training data. The output will be a MultivariateStudentT.
    """

    def __init__(self, train_inputs, train_targets, likelihood):
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        if train_inputs is not None and not all(torch.is_tensor(train_input) for train_input in train_inputs):
            raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")
        if not isinstance(likelihood, StudentTLikelihood):
            raise RuntimeError("ExactTP can only handle Student's t-likelihoods")

        super(ExactTP, self).__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.prediction_strategy = None

    @property
    def train_targets(self):
        return self._train_targets

    @train_targets.setter
    def train_targets(self, value):
        object.__setattr__(self, "_train_targets", value)

    def _apply(self, fn):
        if self.train_inputs is not None:
            self.train_inputs = tuple(fn(train_input) for train_input in self.train_inputs)
            self.train_targets = fn(self.train_targets)
        return super(ExactTP, self)._apply(fn)

    def _clear_cache(self):
        # The precomputed caches from test time live in prediction_strategy
        self.prediction_strategy = None

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If True, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = f"Cannot modify {attr} of inputs (expected {expected_attr}, found {found_attr})."
                            raise RuntimeError(msg)
            self.train_inputs = inputs
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = f"Cannot modify {attr} of targets (expected {expected_attr}, found {found_attr})."
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not all(
                    torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)
                ):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactTP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                # if not isinstance(full_output, MultivariateNormal):
                if not isinstance(full_output, MultivariateStudentT):
                    raise RuntimeError("ExactTP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super().__call__(*train_inputs, **kwargs)

                # Create the prediction strategy for Student-t process
                self.prediction_strategy = StudentTPredictionStrategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training/test data
            full_output = super(ExactTP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                # if not isinstance(full_output, MultivariateNormal):
                if not isinstance(full_output, MultivariateStudentT):
                    raise RuntimeError("ExactTP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction using Student-t process formulas
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                predictive_mean, predictive_covar = self.prediction_strategy.exact_predictive(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()

            # Create a MultivariateStudentT distribution for the predictive posterior
            predictive_df = self.likelihood.df + self.prediction_strategy.num_train
            predictive_dist = MultivariateStudentT(
                loc=predictive_mean, covariance_matrix=predictive_covar, df=predictive_df
            )

            return predictive_dist
