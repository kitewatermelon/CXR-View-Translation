import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from datetime import datetime

from model.discriminator import Discriminator
from model.generator import Generator
from utils.visualization import save_image, save_loss

class Trainer():
    def __init__(self, args, save_path, train_loader, val_loader):
        self.device = args.device
        self.discriminator = Discriminator().to(self.device)
        self.generator = Generator().to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = args.epochs
        self.patience = args.patience
        self.L1_lambda = args.L1_lambda
        self.loss_comparison = nn.BCEWithLogitsLoss() 
        self.L1_loss = nn.L1Loss()
        
        self.discriminator_opt = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.generator_opt = optim.Adam(self.generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.mode = args.mode
        self.save_path = save_path

    def train(self):
        self.patience = 10  
        best_val_loss = float('inf')
        early_stop_counter = 0

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            print(f"Training epoch {epoch + 1}")
            self.generator.train()
            self.discriminator.train()
            train_loss = 0
            for batch in tqdm(self.train_loader):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                Disc_Loss = self.discriminator_training(inputs, targets, self.discriminator_opt)

                for _ in range(2):
                    Gen_Loss, generator_image = self.generator_training(inputs, targets, self.generator_opt, self.L1_lambda)
                    train_loss += Gen_Loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            history['train_loss'].append(avg_train_loss)

            print('Validation phase')
            self.generator.eval()
            self.discriminator.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(self.val_loader):
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)

                    Disc_Loss = self.discriminator_validation(inputs, targets)
                    Gen_Loss, generator_image = self.generator_validation(inputs, targets, self.L1_lambda)
                    val_loss += Gen_Loss.item()

            avg_val_loss = val_loss / len(self.val_loader)
            history['val_loss'].append(avg_val_loss)
            print(f"[INFO] Validation Loss after epoch {epoch + 1}: {avg_val_loss}")

            if epoch > 28:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0 
                    torch.save(self.generator.state_dict(), f'{self.save_path}/model/best_generator.pth')
                    torch.save(self.discriminator.state_dict(), f'{self.save_path}/model/best_discriminator.pth')
                    print(f"[INFO] New best model found and saved at epoch {epoch + 1}")
                else:
                    early_stop_counter += 1
                    print(f"[INFO] No improvement for {early_stop_counter} epoch(s).")

                if early_stop_counter >= self.patience:
                    print("[INFO] Early stopping triggered.")
                    break
            
            save_image(inputs, generator_image, targets, num_images=5,i=1, save_path = self.save_path)

        save_loss(history, self.save_path)
        torch.save(self.generator.state_dict(), f"{self.save_path}/model/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{self.save_path}/model/discriminator.pth")
        print("[INFO] Model has saved")

    def discriminator_training(self,inputs,targets,discriminator_opt):
        
        discriminator_opt.zero_grad()

        # Passing real image and getting loss
        output = self.discriminator(inputs, targets) 
        label = torch.ones(size = output.shape, dtype=torch.float, device=self.device)

        real_loss = self.loss_comparison(output, label) 

        # Passing Generated image and getting loss
        gen_image = self.generator(inputs).detach()

        fake_output = self.discriminator(inputs, gen_image)
        fake_label = torch.zeros(size = fake_output.shape, dtype=torch.float, device=self.device) 

        fake_loss = self.loss_comparison(fake_output, fake_label)

        Total_loss = (real_loss + fake_loss)/2

        Total_loss.backward()

        discriminator_opt.step()

        return Total_loss

    def generator_training(self, inputs,targets, generator_opt, L1_lambda):
            
        generator_opt.zero_grad()
        generated_image = self.generator(inputs)
        
        disc_output = self.discriminator(inputs, generated_image)
        desired_output = torch.ones(size = disc_output.shape, dtype=torch.float, device=self.device)

        generator_loss = self.loss_comparison(disc_output, desired_output) + L1_lambda * self.L1_loss(generated_image,targets)
        generator_loss.backward()
        generator_opt.step()

        return generator_loss, generated_image

    def discriminator_validation(self, inputs, targets):
        """
        Function to evaluate the discriminator during validation or testing.
        """
        with torch.no_grad():  # No gradient computation for validation
            # Passing real image and getting loss
            real_output = self.discriminator(inputs, targets)
            real_label = torch.ones(size=real_output.shape, dtype=torch.float, device=self.device)
            real_loss = self.loss_comparison(real_output, real_label)

            # Passing generated (fake) image and getting loss
            generated_image = self.generator(inputs)
            fake_output = self.discriminator(inputs, generated_image)
            fake_label = torch.zeros(size=fake_output.shape, dtype=torch.float, device=self.device)
            fake_loss = self.loss_comparison(fake_output, fake_label)

            # Total loss for discriminator
            total_loss = (real_loss + fake_loss) / 2
        
        return total_loss

    def generator_validation(self, inputs, targets, L1_lambda):
        """
        Function to evaluate the generator during validation or testing.
        """
        with torch.no_grad():  # No gradient computation for validation
            # Generate images from input
            generated_image = self.generator(inputs)

            # Get the discriminator output for the generated images
            disc_output = self.discriminator(inputs, generated_image)
            desired_output = torch.ones(size=disc_output.shape, dtype=torch.float, device=self.device)

            # Calculate generator loss
            generator_loss = self.loss_comparison(disc_output, desired_output) + L1_lambda * self.L1_loss(generated_image, targets)
        
        return generator_loss, generated_image
