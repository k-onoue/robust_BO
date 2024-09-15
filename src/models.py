from typing import Optional, Union

import gpytorch
import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils import validate_input_scaling
from gpytorch.likelihoods import StudentTLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.module import Module
from torch import Tensor
from tptorch.distributions.multivariate_student_t import MultivariateStudentT
# from tptorch.likelihoods import StudentTLikelihood
from tptorch.models import ExactTP


class SingleTaskTP(BatchedMultiOutputGPyTorchModel, ExactTP, FantasizeMixin):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Module] = None,
        nu: float = 5.0,
        outcome_transform: Optional[Union[OutcomeTransform, str]] = Standardize(m=1),
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        # Initialize the parent classes
        likelihood = StudentTLikelihood()
        ExactTP.__init__(self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood)
        BatchedMultiOutputGPyTorchModel.__init__(self)
        FantasizeMixin.__init__(self)

        # Validate and transform input/output
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        if outcome_transform == Standardize:
            outcome_transform = Standardize(m=train_Y.shape[-1], batch_shape=train_X.shape[:-2])
        transformed_X = self.transform_inputs(X=train_X, input_transform=input_transform)
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        # Set custom parameters
        self.nu = torch.nn.Parameter(torch.tensor(nu))
        self.data_num = torch.tensor(train_Y.shape[0], dtype=torch.float64)

        # Likelihood
        self.likelihood = likelihood

        # Set up the mean and covariance modules
        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or gpytorch.kernels.RBFKernel()

        # Outcome and input transforms
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

        # Ensure the model is on the same device as the training data
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateStudentT:
        """The forward method for the TP model, returning a Student's t distribution."""
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x_train_data = self.covar_module(self.train_inputs[0])
        inv_quad, _ = covar_x_train_data.inv_quad_logdet(
            inv_quad_rhs=self.train_targets - self.train_targets.mean(), logdet=False
        )

        tp_var_scale = (self.nu + inv_quad - 2) / (self.nu + self.data_num - 2)
        covar_x = tp_var_scale * covar_x

        return MultivariateStudentT(mean_x, covar_x, self.nu, self.data_num)



if __name__ == "__main__":
    import warnings
    import torch
    from test_functions import sinusoidal_synthetic

    warnings.filterwarnings("ignore")

    test_function = sinusoidal_synthetic
    # Generate input-output pairs from the test function
    n_train = 30
    train_X = torch.linspace(-1, 1, n_train).view(-1, 1)
    train_Y = test_function(train_X) + torch.randn(train_X.size()) * 0.5  # Add noise


    from tptorch.mlls.exact_student_t_marginal_log_likelihood import ExactStudentTMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior


    # Define the Matern Kernel with nu=2.5 and ARD enabled
    ard_num_dims = train_X.shape[-1]  # Input dimension for ARD
    matern_kernel = MaternKernel(
        nu=2.5, ard_num_dims=ard_num_dims, lengthscale_prior=GammaPrior(3.0, 6.0)
    )

    # ScaleKernel allows for automatic scaling of the output
    kernel = ScaleKernel(matern_kernel, outputscale_prior=GammaPrior(2.0, 0.15))

    # Define the Student-t Process (TP) model using the specified kernel
    model = SingleTaskTP(train_X, train_Y, covar_module=kernel)
    mll = ExactStudentTMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)