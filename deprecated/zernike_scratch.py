class Zernike(nn.Module):
    def __init__(self, n_max: int, m_max: int, npixels: int):
        super().__init__()
        self.n_max = n_max
        self.m_max = m_max
        self.npixels = npixels
        number_of_zernikes = (n_max + 1) * (n_max + 2) // 2
        self.basis_function_weights = nn.Parameter(
            torch.randn(number_of_zernikes, dtype=torch.float64), requires_grad=True
        )
        self.center_x = nn.Parameter(
            torch.tensor(npixels / 2, dtype=torch.float64), requires_grad=False
        )
        self.center_y = nn.Parameter(
            torch.tensor(npixels / 2, dtype=torch.float64), requires_grad=False
        )

    @staticmethod
    def _factorial(x: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(x + 1).exp()

    def _f_nms(self, n: int, m: int, s: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        s_3d = s.view(-1, 1, 1)
        return (
            (-1) ** s_3d
            * self._factorial(n - s_3d)
            * self._factorial((n - m) / 2 - s_3d)
            * rho ** (n - 2 * s_3d)
            / (self._factorial(s_3d) * self._factorial((n + m) / 2 - s_3d))
        )

    def _summation(self, n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
        indices_s = torch.arange(0, (n - m) / 2, dtype=torch.float64, device=rho.device)
        return self._f_nms(n, m, indices_s, rho).sum(dim=0)

    def _R_nm(self, n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
        radial_contribution = torch.zeros_like(rho, dtype=torch.float64)
        radial_contribution += self._summation(n, m, rho)
        return radial_contribution

    def sum_basis_funcs(self):
        device = self.center_x.device
        x = torch.linspace(0, self.npixels - 1, self.npixels, dtype=torch.float64, device=device)
        y = torch.linspace(0, self.npixels - 1, self.npixels, dtype=torch.float64, device=device)
        x, y = torch.meshgrid(x, y, indexing="xy")
        x = x - self.center_x
        y = y - self.center_y
        rho = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        idx = 0
        output = torch.zeros((self.npixels, self.npixels), dtype=torch.float64,device=device)
        ones = torch.ones((self.npixels, self.npixels), dtype=torch.float64, device=device)

        for n in range(self.n_max + 1):
            for m in range(n + 1):
                if (n - m) % 2 == 0:
                    if m == 0:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.sqrt(torch.tensor(n + 1, dtype=torch.float32, device=output.device))
                        )
                        idx += 1
                    elif m % 2 == 0:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.cos(m * theta)
                            * torch.sqrt(torch.tensor(2*(n + 1), dtype=torch.float32, device=output.device))
                        )
                        idx += 1
                    else:
                        output += (
                            self.basis_function_weights[idx]
                            * self._R_nm(n, m, rho)
                            * torch.sin(m * theta)
                            * torch.sqrt(torch.tensor(2*(n + 1), dtype=torch.float32, device=output.device))
                        )
                        idx += 1
        
        rescaled = output + ones
        normalization_constant = torch.max(rescaled)
        return rescaled / normalization_constant

