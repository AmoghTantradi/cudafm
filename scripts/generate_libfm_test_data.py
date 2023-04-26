import numpy as np
import random
import argparse

"""
Example Usage:
```py
python generate_libfm_test_data.py -n 100 -f 50 -s 0.1 -o libfm_test_data.txt
```
or on perlmutter:
```py
python3 generate_libfm_test_data.py -n 200000 -f 50 -s 0.4 -o libfm_test_data.txt
```
"""

np.random.seed(0)

def generate_libfm_data(num_samples, num_features, sparsity, output_file):
    samp_size = int(num_features * sparsity)
    
    # Generate X and y
    samp_indices = np.random.choice(num_features, (num_samples, samp_size), replace=True)
    random_values = np.random.uniform(-10, 10, (num_samples, samp_size))
    X = np.zeros((num_samples, num_features))
    X[np.arange(num_samples)[:, None], samp_indices] = random_values
    y = X[:, 0]
    
    # Save to file
    with open(output_file, 'w') as f:
        for i in range(num_samples):
            f.write(f'{y[i]:.2f}')
            nonzero_indices = np.nonzero(X[i, :])[0]
            for j in nonzero_indices:
                f.write(f' {j}:{X[i, j]:.2f}')
            f.write('\n')
    # X = np.zeros((num_samples, num_features))
    # # y = np.random.randint(-10, 11, num_samples)
    # y = np.ones(num_samples)
    # # quadratic function to learn:
    # g = lambda x: x ** 2
    # g2 = lambda x: 10.1 * x**2 + 4.2 * x + 1.3

    # # generate a poisson distribution of number of non-zero features per sample
    # # num_non_zero_features = np.random.poisson(num_features * sparsity, num_samples)

    # for i in range(num_sample):
    #     samp = np.random.choice(num_features, int(num_features * sparsity), replace=False)
    #     X[i][samp] = np.random.uniform(-10, 10, int(num_features * sparsity))

    # for i in range(num_samples):
    #     y[i] += g(X[i, 0])

    # with open(output_file, 'w') as f:
    #     for i in range(num_samples):
    #         f.write(f'{y[i]:.5f}')
    #         for j in range(num_features):
    #             if X[i, j] != 0:
    #                 f.write(f' {j}:{X[i, j]:.2f}')
    #         f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate libFM test data.')
    parser.add_argument('-n', '--num_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('-f', '--num_features', type=int, required=True, help='Number of features')
    parser.add_argument('-s', '--sparsity', type=float, required=True, help='Sparsity of the data (0 to 1)')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file in libSVM format')

    args = parser.parse_args()
    generate_libfm_data(args.num_samples, args.num_features, args.sparsity, args.output_file)
