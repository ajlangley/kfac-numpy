import numpy as np
import scipy.linalg as la

class KFAC:
    def __init__(self, mlp, approx='diagonal', eta=1e-5, lam=150, sample_ratio=0.25, update_approx_every=20,
                update_gamma_every=20, T_2=20):
        '''
            sample_ratio is the proportion of the input batch to use when computing the second backward pass
                wit
        '''
        self.mlp = mlp
        # Keep references to the linear layers for easy access
        self.lin_layers = [layer for layer in mlp.layers if type(layer) is Layer]
        # Generate some other statistics about the model to be optimized
        self.n_layers = len(self.lin_layers)
        self.n_params = np.int64(np.sum([layer.dW.size for layer in self.lin_layers]))
        self.sample_ratio = sample_ratio
        self.update_approx_every = update_approx_every
        self.update_gamma_every = update_gamma_every

        self.A = {}
        self.G = {}
        self.A_inv_damp = {}
        self.G_inv_damp = {}
        self.grads_flat = np.empty((self.n_params, 1))
        self.lam = lam
        self.eta = eta
        self.gamma = np.sqrt(lam + eta)
        self.omega = np.sqrt(19 / 20) ** T_2
        self.gamma_coef = np.asarray([1, self.omega, 1 / self.omega])
        self.n_iter = 1
        self.approx = approx

        # Precompute coordinates for the block entries needed for the Fisher approximation
        if approx == 'diagonal':
            self.block_coords = list(zip(np.arange(self.n_layers), np.arange(self.n_layers)))

    def step(self, loss):
        eps = min(1 - 1 / self.n_iter, 0.95)
        A_t, G_t = self.calc_stats()
        # Update A and G
        eps = min(1 - 1 / self.n_iter, 0.95)
        for coord in self.block_coords:
            if self.n_iter == 1:
                self.A[coord] = A_t[coord]
                self.G[coord] = G_t[coord]
            else:
                np.add(np.multiply(eps, self.A[coord]), np.multiply(1 - eps, A_t[coord]), out=self.A[coord])
                np.add(np.multiply(eps, self.G[coord]), np.multiply(1 - eps, G_t[coord]), out=self.G[coord])

        gammas = self.get_candidate_gammas()
        M_min = np.inf
        for gamma in gammas:
            if not self.n_iter % self.update_approx_every or self.n_iter < 4:
                # Recompute the inverse blocks
                for coord in self.block_coords:
                    self.A_inv_damp[coord], self.G_inv_damp[coord] = self.calc_inverse_stats(self.A[coord],
                                                                                   self.G[coord],
                                                                                   gamma)

            update_proposal = self.approx_fisher_prod()
            alpha, M = self.calc_quad_approx_params(update_proposal, loss)

            if M < M_min:
                opt_gamma = gamma
                opt_alpha = alpha
                opt_proposal = update_proposal
                M_min = M

        self.gamma = opt_gamma
        update = opt_alpha * opt_proposal
        # Momentum goes here if time
        self.update_mlp_params(update)
        self.n_iter += 1

    def get_candidate_gammas(self):
        if not self.n_iter % self.update_gamma_every or self.n_iter == 1:
            return self.gamma * self.gamma_coef

        return [self.gamma]

    # Calculate the Jacobian of the output of the last linear layer w.r.t. all of the model parameters
    def calc_jacobian(self):
        J_list = []
        preacts = self.lin_layers[-1].a

        # Calculate Jacobian for last layer
        J_W = np.zeros((*preacts.shape, *self.lin_layers[-1].W.shape))
        # I'll vectorize this later
        for i in range(J_W.shape[0]):
            for j in range(J_W.shape[1]):
                J_W[i, j, j] = self.mlp.layers[-1].x[i]
        J_W = J_W.reshape(J_W.shape[0], J_W.shape[1], -1)
        J_list.append(J_W)

        # Calculate Jacobian for the second to last layer
        if len(self.lin_layers) > 1:
            J_z = self.lin_layers[-1].W
            J_a = np.multiply(J_z[:, :-1], np.expand_dims(ReLU_grad(self.mlp.layers[-2].a), 1))
            J_W = np.einsum('ijk,il->ijkl', J_a, self.mlp.layers[-3].x)
            J_W = J_W.reshape(J_W.shape[0], J_W.shape[1], -1)
            J_list.append(J_W)

        # Calculate Jacobian for the rest of the layers
        for i, j in zip(np.arange(len(self.lin_layers))[-3::-1], np.arange(self.mlp.n_layers)[-4:0:-2]):
            J_z = self.lin_layers[i + 1].W
            J_a = np.multiply(np.matmul(J_a, J_z)[..., :-1], np.expand_dims(ReLU_grad(self.mlp.layers[j].a), 1))
            J_W = np.einsum('ijk,il->ijkl', J_a, self.mlp.layers[j - 1].x)
            J_W = J_W.reshape(J_W.shape[0], J_W.shape[1], -1)
            J_list.append(J_W)

        J_list.reverse()
        J = np.concatenate(J_list, axis=-1)

        return J

    def approx_fisher_prod(self):
        fisher_prod = np.empty(self.grads_flat.shape)
        i = 0

        if self.approx == 'diagonal':
            for coord in self.block_coords:
                W = self.lin_layers[coord[0]].W
                V = self.grads_flat[i:i + W.size].reshape(W.shape)
                U = np.matmul(self.G_inv_damp[coord], np.matmul(-V, self.A_inv_damp[coord]))
                fisher_prod[i:i + W.size] = U.reshape(-1, 1)
                i += W.size

        return fisher_prod

    def exact_fisher_prod(self, proposal):
        batch_size = self.lin_layers[0].x.shape[0]

        J = self.calc_jacobian()

        F_prod = 0
        for i in range(batch_size):
            J_prod = np.matmul(J[i], proposal)
            p = self.mlp.g[i,:]
            q = np.sqrt(p)
            B = np.diag(q) - np.outer(p, q)
            F_R = np.matmul(B.T, B)
            F_prod += np.matmul(J_prod.T, np.matmul(F_R, J_prod))

        return F_prod / batch_size

    def calc_quad_approx_params(self, proposal, loss):
        grads_dot_proposal = np.dot(-self.grads_flat.ravel(), proposal.ravel())
        exact_fisher_prod = self.exact_fisher_prod(proposal) + (self.lam + self.eta) * la.norm(proposal)
        alpha = grads_dot_proposal / (exact_fisher_prod)
        M = alpha ** 2 / 2 + alpha * grads_dot_proposal + loss

        return alpha, M

    def update_mlp_params(self, update):
        i = 0
        for layer in self.lin_layers:
            layer_size = layer.W.size
            layer.W += update[i:i + layer_size].reshape(layer.W.shape)
            i += layer_size

    # See sections 3 and 5 of (Martens and Grosse 2015). Compute the stats needed about the
    # current state of the model to create the update proposal
    def calc_stats(self):
        batch_size = self.mlp.g.shape[0]
        sample_size = int(batch_size * self.sample_ratio)
        A = {}
        G = {}

        # Get flattened version of all of the model's gradients concatenated together
        i = 0
        for layer in self.lin_layers:
            np.copyto(self.grads_flat[i:i + layer.W.size], layer.dW.reshape(-1, 1))
            i += layer.W.size
        self.grads_flat /= batch_size

        # Get A's and compute expectation over the batch
        a = [layer.x for layer in self.lin_layers]
        for coord in self.block_coords:
            A[coord] = np.matmul(a[coord[0]].T, a[coord[1]]) / batch_size

        # Perform the second backprop with targets sampled from the predictive distribution
        # Pick the distributions in the batch to sample from randomly
        start_rand = np.random.randint(0, batch_size - sample_size, 1)[0]
        end_rand = start_rand + sample_size
        rand_slice = slice(start_rand, end_rand)
        sample_targets = self.sample_outputs(self.mlp.g[rand_slice])
        g = []
        # Get the G's and compute the expectation over the samples drawn from the output distribution
        # g corresponds to dL/ds using the paper's notation (s is the preactivation value)
        delta = sample_targets - self.mlp.g[rand_slice]
        for layer in self.mlp.layers[::-1]:
            if type(layer) is Layer:
                g.append(delta)
                dx = np.matmul(delta, layer.W[:, :-1])
            else:
                delta = np.multiply(layer.grad(layer.a[rand_slice]), dx)

        g.reverse()
        for coord in self.block_coords:
            G[coord] = np.matmul(g[coord[0]].T, g[coord[1]]) / sample_size

        return A, G

    # Calculate a block of the approximate inverse Fisher with damping. See section 6.3 of paper
    def calc_inverse_stats(self, A, G, gamma):
        pi_i = self.trace_norm(A, G)
        A_damp_inv = la.inv(np.add(A, pi_i * gamma * np.eye(A.shape[0])))
        G_damp_inv = la.inv(np.add(G, 1 / pi_i * gamma * np.eye(G.shape[0])))

        return A_damp_inv, G_damp_inv

    def trace_norm(self, A, G):
        A_tr, G_tr = np.trace(A), np.trace(G)
        tr_norm = np.sqrt((A_tr / (A.shape[0])) / (G_tr / G.shape[0]))

        return tr_norm

    def sample(self, g):
        sample = np.argmax(np.random.multinomial(1, g, size=1))

        return sample

    def sample_outputs(self, g):
        return np.expand_dims(np.apply_along_axis(self.sample, axis=1, arr=g), 1)
