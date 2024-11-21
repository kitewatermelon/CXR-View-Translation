import torch 
import torch.nn as nn
import pandas as pd

from model.discriminator import Discriminator
from model.generator import Generator
from utils.visualization import save_image, save_loss
from metric.metric import *

class Tester():
    def __init__(self, args, save_path, test_loader):
        self.device = args.device
        self.discriminator = Discriminator().to(self.device)
        self.generator = Generator().to(self.device)
        self.test_loader = test_loader
        self.loss_comparison = nn.BCEWithLogitsLoss()
        self.L1_loss = nn.L1Loss()
        self.save_path = save_path
        self.mode = args.mode
    def test(self):
        self.generator.eval()
        self.discriminator.eval()
        
        test_loss = 0
        discriminator_losses = []
        generator_losses = []
        psnrs = []
        ssims = []
        mses = []

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                # Discriminator Test
                Disc_Loss = self.discriminator_validation(inputs, targets)
                discriminator_losses.append(Disc_Loss.item())

                # Generator Test
                Gen_Loss, generator_image = self.generator_validation(inputs, targets)
                generator_losses.append(Gen_Loss.item())
                test_loss += Gen_Loss.item()

                # Metrics: PSNR, SSIM, MSE
                psnr_value = compute_psnr(targets, generator_image)
                psnrs.append(psnr_value)

                ssim_value = compute_ssim(targets, generator_image)
                ssims.append(ssim_value)

                mse_value = compute_mse(targets, generator_image)
                mses.append(mse_value)

                # Save images
                save_image(inputs, generator_image, targets, num_images=5, i=i, save_path=self.save_path, is_train=False)

        avg_test_loss = test_loss / len(self.test_loader)
        avg_discriminator_loss = np.mean(discriminator_losses)
        avg_generator_loss = np.mean(generator_losses)
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        avg_mse = np.mean(mses)

        # Save results to CSV
        results_df = pd.DataFrame({
            'Batch': range(1, len(generator_losses) + 1),
            'Discriminator_Loss': discriminator_losses,
            'Generator_Loss': generator_losses,
            'PSNR': psnrs,
            'SSIM': ssims,
            'MSE': mses
        })

        # Add average results
        avg_results_df = pd.DataFrame({
            'Batch': ['Average'],
            'Discriminator_Loss': [avg_discriminator_loss],
            'Generator_Loss': [avg_generator_loss],
            'PSNR': [avg_psnr],
            'SSIM': [avg_ssim],
            'MSE': [avg_mse]
        })

        results_df = pd.concat([results_df, avg_results_df], ignore_index=True)

        # Save to CSV
        results_df.to_csv(f'{self.save_path}/metrics/metrics.csv', index=False)
        print("[INFO] Test results saved to CSV.")

    def discriminator_validation(self, inputs, targets):
        with torch.no_grad():
            real_output = self.discriminator(inputs, targets)
            real_label = torch.ones(size=real_output.shape, dtype=torch.float, device=self.device)
            real_loss = self.loss_comparison(real_output, real_label)

            generated_image = self.generator(inputs)
            fake_output = self.discriminator(inputs, generated_image)
            fake_label = torch.zeros(size=fake_output.shape, dtype=torch.float, device=self.device)
            fake_loss = self.loss_comparison(fake_output, fake_label)

            total_loss = (real_loss + fake_loss) / 2
        
        return total_loss

    def generator_validation(self, inputs, targets):
        with torch.no_grad():
            generated_image = self.generator(inputs)
            disc_output = self.discriminator(inputs, generated_image)
            desired_output = torch.ones(size=disc_output.shape, dtype=torch.float, device=self.device)

            generator_loss = self.loss_comparison(disc_output, desired_output) + self.L1_loss(generated_image, targets)
        
        return generator_loss, generated_image
