import dataclasses
import types

import accelerate
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class VanillaGAN:
    cfg: types.SimpleNamespace
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    discriminator_opt: torch.optim.Optimizer
    discriminator_scheduler: torch.optim.lr_scheduler._LRScheduler
    accelerator: accelerate.Accelerator

    @property
    def _batch_size(self) -> int:
        return self.cfg.bs
    
    def compute_gradient_penalty(self, d_out: torch.Tensor, d_in: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=d_in,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return gradients.pow(2).sum().mean()
    
    def set_discriminator_requires_grad(self, rg: bool) -> None:
        for module in self.discriminator.parameters():
            module.requires_grad = rg

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = (d_real_logits.sigmoid() < 0.5).float().mean().item()
        disc_acc_fake = (d_fake_logits.sigmoid() > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (
                disc_loss + 
                ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)
            ) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits.sigmoid() < 0.5).float().mean().item()
        return gen_loss, gen_acc

    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        if self.cfg.loss_coefficient_disc > 0:
            return self._step_discriminator(real_data, fake_data)
        else:
            return torch.tensor(0.0), 0.0, 0.0

    def step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.cfg.loss_coefficient_gen > 0:
            return self._step_generator(real_data=real_data, fake_data=fake_data)
        else:
            return torch.tensor(0.0), 0.0

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        self.generator.eval()
        self.discriminator.train()
        self.set_discriminator_requires_grad(True)
        r1_penalty, disc_loss, disc_acc_real, disc_acc_fake = self.step_discriminator(
            real_data=real_data.detach(),
            fake_data=fake_data.detach()
        )
        self.generator.train()
        self.discriminator.eval()
        self.set_discriminator_requires_grad(False)
        gen_loss, gen_acc = self.step_generator(
            real_data=real_data,
            fake_data=fake_data
        )

        return r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc



class LeastSquaresGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = (d_real_logits ** 2).mean()
        disc_loss_fake = ((d_fake_logits - 1) ** 2).mean()
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = ((d_real_logits ** 2) < 0.5).float().mean().item()
        disc_acc_fake = ((d_fake_logits ** 2) > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)
        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (disc_loss + ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        gen_loss = ((d_fake_logits) ** 2).mean()
        gen_acc = ((d_fake_logits ** 2) < 0.5).float().mean().item()
        return gen_loss * 0.5, gen_acc


class RelativisticGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)

        disc_loss = F.binary_cross_entropy_with_logits(d_fake_logits - d_real_logits, torch.ones_like(d_real_logits))
        disc_acc_real = (d_real_logits > d_fake_logits).float().mean().item()
        disc_acc_fake = 1.0 - disc_acc_real

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(disc_loss * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()

        return disc_loss, disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()

        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        gen_loss = F.binary_cross_entropy_with_logits(d_real_logits - d_fake_logits, torch.ones_like(d_real_logits))

        gen_acc = (d_real_logits > d_fake_logits).float().mean().item()
        self.discriminator.train()
        return gen_loss, gen_acc


class WassersteinGAN(VanillaGAN):
    """Wasserstein GAN with Gradient Penalty (WGAN-GP).

    Uses Wasserstein distance estimation with gradient penalty
    to enforce Lipschitz constraint on the critic.
    """

    def compute_wgan_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP.

        Interpolates between real and fake samples, computes critic output,
        and penalizes deviation of gradient norm from 1.
        """
        batch_size = real_data.size(0)
        device = real_data.device

        # Random interpolation coefficient
        epsilon = torch.rand(batch_size, 1, device=device)

        # Interpolate between real and fake
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated = interpolated.requires_grad_(True)

        # Get critic output for interpolated samples
        d_interpolated = self.discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute gradient penalty: (||grad||_2 - 1)^2
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """WGAN critic update step.

        Critic loss = E[D(fake)] - E[D(real)] + lambda_gp * gradient_penalty
        """
        real_data = real_data.detach()
        fake_data = fake_data.detach()

        # Critic outputs (no sigmoid - raw scores)
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data)

        # Wasserstein distance estimate (negative because we want to maximize)
        # Critic wants: D(real) high, D(fake) low
        # So critic loss = E[D(fake)] - E[D(real)]
        wasserstein_dist = d_real.mean() - d_fake.mean()
        critic_loss = -wasserstein_dist  # Minimize negative = maximize distance

        # Gradient penalty
        gp_lambda = getattr(self.cfg, 'gp_lambda', 10.0)  # Default lambda=10
        gradient_penalty = self.compute_wgan_gradient_penalty(real_data, fake_data)

        # Total critic loss
        total_critic_loss = critic_loss + gp_lambda * gradient_penalty

        # "Accuracy" metrics (for compatibility with logging)
        # In WGAN, we use the sign of the output as a proxy
        disc_acc_real = (d_real > 0).float().mean().item()
        disc_acc_fake = (d_fake < 0).float().mean().item()

        # Backward pass
        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(total_critic_loss * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()

        return gradient_penalty.detach(), critic_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        """WGAN generator update step.

        Generator loss = -E[D(fake)]
        Generator wants critic to think fake samples are real (high scores).
        """
        # Get critic score for fake samples
        d_fake = self.discriminator(fake_data)

        # Generator wants to maximize D(fake), so minimize -D(fake)
        gen_loss = -d_fake.mean()

        # "Accuracy" metric (proxy)
        gen_acc = (d_fake > 0).float().mean().item()

        return gen_loss, gen_acc
