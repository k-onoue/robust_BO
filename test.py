# import math
# import torch
# import gpytorch
# import matplotlib.pyplot as plt
# from gpytorch import constraints  # Corrected import

# # Import your custom classes
# from src.exact_tp import ExactTP
# from src.student_t_likelihood import StudentTLikelihood
# from src.exact_student_t_marginal_log_likelihood import ExactStudentTMarginalLogLikelihood
# from src.multivariate_student_t import MultivariateStudentT

# def test_exact_tp():
#     # Step 1: Generate synthetic data with heavy-tailed noise
#     train_x = torch.linspace(0, 1, 100)
#     true_function = lambda x: torch.sin(x * (2 * math.pi))
#     torch.manual_seed(0)
#     df_noise = 3.0  # Degrees of freedom for the noise
#     noise = torch.distributions.StudentT(df=df_noise).rsample(train_x.shape)
#     train_y = true_function(train_x) + 0.2 * noise

#     # Step 2: Define the StudentTGPModel class that inherits from ExactTP
#     class StudentTGPModel(ExactTP):
#         def __init__(self, train_x, train_y, likelihood):
#             super(StudentTGPModel, self).__init__(train_x, train_y, likelihood)
#             self.mean_module = gpytorch.means.ZeroMean()
#             self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#             self.df = likelihood.df

#         def forward(self, x):
#             mean_x = self.mean_module(x)
#             covar_x = self.covar_module(x)
#             return MultivariateStudentT(
#                 loc=mean_x, covariance_matrix=covar_x, df=self.df
#             )

#     # Initialize the Student's t-likelihood and the StudentTGPModel
#     likelihood = StudentTLikelihood(
#         df=3.0,
#         df_constraint=constraints.Interval(2.01, 1000.0)
#     )
#     model = StudentTGPModel(train_x, train_y, likelihood)

#     # Freeze the df parameter
#     # likelihood.raw_df.requires_grad = False

#     # Print initial parameter values
#     print(f'Initial DF: {likelihood.df.item():.3f}')
#     print(f'Initial Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}')
#     print(f'Initial Noise: {likelihood.noise.item():.3f}')

#     # Step 3: Train the model
#     model.train()
#     likelihood.train()

#     # Use the Adam optimizer
#     optimizer = torch.optim.Adam([
#         {'params': model.parameters()},
#         {'params': likelihood.noise_covar.parameters()},
#     ], lr=0.1)

#     # Define the loss function (negative log marginal likelihood)
#     mll = ExactStudentTMarginalLogLikelihood(likelihood, model)

#     training_iterations = 100
#     for i in range(training_iterations):
#         optimizer.zero_grad()
#         output = model(train_x)
#         loss = -mll(output, train_y)
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % 10 == 0 or i == 0:
#             print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f} - '
#                   f'Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f} - '
#                   f'DF: {likelihood.df.item():.3f} - Noise: {likelihood.noise.item():.3f}')

#     # Step 4: Switch to evaluation mode and make predictions
#     model.eval()
#     likelihood.eval()

#     # Test points are evenly spaced along [0,1]
#     test_x = torch.linspace(0, 1, 200)

#     # Make predictions
#     with torch.no_grad():
#         observed_pred = likelihood(model(test_x))

#     # Step 5: Plot the results
#     with torch.no_grad():
#         # Get lower and upper confidence bounds
#         lower, upper = observed_pred.confidence_region()
#         # Plot training data as black stars
#         plt.figure(figsize=(12, 6))
#         plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Training Data')
#         # Plot predictive mean as blue line
#         plt.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Predictive Mean')
#         # Shade between the lower and upper confidence bounds
#         plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence Interval')
#         # Plot the true function as a red dashed line
#         plt.plot(test_x.numpy(), true_function(test_x).numpy(), 'r--', label='True Function')
#         plt.legend()
#         plt.title("Student-t Process Regression with Student's t-Likelihood")
#         plt.show()

# if __name__ == '__main__':
#     test_exact_tp()


import math
import torch
import gpytorch
import matplotlib.pyplot as plt
from gpytorch import constraints

# Import your custom classes
from src.exact_tp import ExactTP
from src.student_t_likelihood import StudentTLikelihood
from src.exact_student_t_marginal_log_likelihood import ExactStudentTMarginalLogLikelihood
from src.multivariate_student_t import MultivariateStudentT

def test_exact_tp():
    # Step 1: Generate synthetic data with heavy-tailed noise
    train_x = torch.linspace(0, 1, 100)
    true_function = lambda x: torch.sin(x * (2 * math.pi))
    torch.manual_seed(0)
    df_noise = 3.0  # Degrees of freedom for the noise
    noise = torch.distributions.Normal(loc=0,scale=1).rsample(train_x.shape)
    train_y = true_function(train_x) + 0.2 * noise

    # Step 2: Define the StudentTGPModel class that inherits from ExactTP
    class StudentTGPModel(ExactTP):
        def __init__(self, train_x, train_y, likelihood):
            super(StudentTGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            df = likelihood.df
            return MultivariateStudentT(
                loc=mean_x, covariance_matrix=covar_x, df=df
            )

    # Initialize the Student's t-likelihood and the StudentTGPModel
    likelihood = StudentTLikelihood(
        df=3.0,
        df_constraint=constraints.Interval(2.01, 1000.0)
    )
    # Properly initialize raw_df to correspond to df=3.0
    likelihood.raw_df.data = likelihood.raw_df_constraint.inverse_transform(torch.tensor(3.0))
    # Freeze the df parameter
    likelihood.raw_df.requires_grad = True

    model = StudentTGPModel(train_x, train_y, likelihood)

    # Print initial parameter values
    print(f'Initial DF: {likelihood.df.item():.3f}')
    print(f'Initial Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}')
    print(f'Initial Noise: {likelihood.noise.item():.3f}')

    # Step 3: Train the model
    model.train()
    likelihood.train()

    print()
    for name, param in model.named_parameters():
        print(f"Model {name}: {param}")
    for name, param in likelihood.named_parameters():
        print(f"Likelihood {name}: {param}")
    print()


    # Use the Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.noise_covar.parameters()},
    ], lr=0.1)

    

    # Define the loss function (negative log marginal likelihood)
    mll = ExactStudentTMarginalLogLikelihood(likelihood, model)

    training_iterations = 1000
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0 or i == 0:
            print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f} - '
                  f'Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f} - '
                  f'DF: {likelihood.df.item():.3f} - Noise: {likelihood.noise.item():.3f}')

    # Step 4: Switch to evaluation mode and make predictions
    model.eval()
    likelihood.eval()

    # Test points are evenly spaced along [0,1]
    test_x = torch.linspace(0, 1, 200)

    # Make predictions
    with torch.no_grad():
        observed_pred = likelihood(model(test_x))

        # Calculate standard deviation from the covariance matrix
        std_devs = observed_pred.covariance_matrix.diag().sqrt()
        # Define the multiplier for the confidence region (e.g., 2 corresponds to 95% confidence)
        multiplier = 2.0
        lower = observed_pred.mean - multiplier * std_devs
        upper = observed_pred.mean + multiplier * std_devs

    # print(observed_pred.loc)


    # Step 5: Plot the results
    with torch.no_grad():
        # Plot training data as black stars
        plt.figure(figsize=(12, 6))
        plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Training Data')
        # Plot predictive mean as blue line
        plt.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Predictive Mean')
        # Shade between the lower and upper confidence bounds
        plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence Interval')
        # Plot the true function as a red dashed line
        plt.plot(test_x.numpy(), true_function(test_x).numpy(), 'r--', label='True Function')
        plt.legend()
        plt.title("Student-t Process Regression with Student's t-Likelihood")
        plt.show()

if __name__ == '__main__':
    test_exact_tp()
