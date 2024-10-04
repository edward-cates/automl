import torch
import numpy as np
from tqdm.auto import tqdm

from src.user_io.user_io_cache import UserIoCache
from src.llm.chatgpt import ToolDescriptor, ToolArgument

class UserIoCacheProjectSpecific(UserIoCache):
    """
    Extend the properties, descriptors, methods
    """

    train_losses: list[float] = list()
    test_losses: list[float] = list()

    @property
    def vae(self) -> torch.nn.Module:
        return self.models["tiny_autoencoder"]

    @property
    def diffuser(self) -> torch.nn.Module:
        return self.models["tiny_diffuser"]

    @property
    def run_train_epoch_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="run_train_epoch",
            description="Run a training epoch.",
        )

    def run_train_epoch(self) -> str:
        for model in self.models.values():
            model.train()

        total_losses = []
        image_losses = []
        noise_losses = []

        for x in tqdm(self.train_dataloader, desc="Training"):
            img_loss, noise_loss = self.step_model(x.to(self.device))
            total_loss = img_loss + noise_loss

            image_losses.append(img_loss.item())
            noise_losses.append(noise_loss.item())
            total_losses.append(total_loss.item())

            total_loss.backward()
            # clip gradients.
            for model in self.models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.train_losses = np.array(total_losses).mean()

        return f"Training complete. Total loss: {total_losses[-1]}, Image loss: {image_losses[-1]}, Noise loss: {noise_losses[-1]}"

    def step_model(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (Original).
        1. Encode.
        2. Create noise. -> Noise.
        3. Add to latent and Predict noise. -> Predicted noise.
        4. Denoise.
        5. Decode. -> Decoded.
        Loss = MSE(Original., Decoded) + MSE(Noise, Predicted Noise)
        """
        latent = self.vae.encode(x)
        latent_noisy, t, noise = self.diffuser.create_noised_image(latent)
        predicted_noise = self.diffuser(latent_noisy)
        actual_noise = latent_noisy - latent
        latent_0 = latent_noisy - predicted_noise
        x_0 = self.vae.decode(latent_0)
        x_ae = self.vae.decode(latent)
        ae_loss = torch.nn.functional.mse_loss(x_ae, x, reduction='mean')
        ae_transition_loss = self._calc_transition_loss(x_ae, x)
        img_loss = torch.nn.functional.mse_loss(x_0, x, reduction='mean')
        img_transition_loss = self._calc_transition_loss(x_0, x)
        # 100 is observed to be a good scaling factor.
        img_loss = (img_loss + img_transition_loss + ae_loss + ae_transition_loss) / 4.0 * 50.0
        noise_loss = torch.nn.functional.mse_loss(predicted_noise, actual_noise, reduction='mean')
        return img_loss, noise_loss

    @staticmethod
    def _calc_transition_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = UserIoCacheProjectSpecific._calc_transitions(x)
        y = UserIoCacheProjectSpecific._calc_transitions(y)
        return torch.nn.functional.mse_loss(x, y, reduction='mean')

    @staticmethod
    def _calc_transitions(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]
