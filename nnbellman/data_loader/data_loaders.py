from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
from model_loader import load_model
import torch
import pandas as pd
import numpy as np
import sys


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class InformedBellmanDataset(Dataset):
    '''Takes input from a model of a single variable as an input to a model of another variable.'''
    def __init__(self, input_csv_file, output_csv_file, output_variable, output_format, info_model_path, info_model_format="category", categories=None):
        self.input_data = pd.read_csv(input_csv_file)
        self.output_data = pd.read_csv(output_csv_file)
        self.info_data = pd.DataFrame()

        
        #Subset data inputs
        self.input_data = self.input_data[['Alpha','k','Sigma','Theta']].values
        #Eliminate and Track Duplicate Rows
        u , unique_indices = np.unique(self.input_data, axis=0, return_index=True)
        self.input_data = self.input_data[np.sort(unique_indices)]

        if output_variable=="consumption":
            self.output_data = self.output_data[['Consumption']].values
            self.output_data = self.output_data[np.sort(unique_indices)]

            if categories==None:
                print("You have not defined possible_i_a for the data loader. The adaptation costs are essential for the two-step version of consumption prediction.")
                sys.exit()
            else:
                possible_i_a = torch.tensor(list(categories.values()))
            #Load i_a model, predict for input, reformat, append to input
            info_model = load_model(info_model_path)
            info_model.eval() 

            with torch.no_grad():
                info_output = info_model(torch.tensor(self.input_data, dtype=torch.float32))
            if info_model_format=="decimal":
                self.input_data = torch.cat((self.input_data, possible_i_a[torch.argmin(torch.abs(info_output[:,0].unsqueeze(-1) - possible_i_a), dim=-1)].unsqueeze(1)), dim=1).detach().to('cpu').numpy()

            elif info_model_format=="category":
                self.input_data = np.concatenate((self.input_data, possible_i_a[torch.argmax(info_output, dim=1)].unsqueeze(1).detach().to('cpu').numpy()), axis=1)

            else: 
                print('info_model_format must be set to supported format, "decimal" or "category".')
        else: 
            print(f"Informed training not available for output variable{output_variable}.")
            sys.exit()

        


 

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):


        input_sample = torch.tensor(self.input_data[idx], dtype=torch.float32)
        output_sample = torch.tensor(self.output_data[idx], dtype=torch.float32)


        return input_sample, output_sample


class BellmanDataset(Dataset):
    '''Values for "both" output are normalized against the range.'''
    def __init__(self, input_csv_file, output_csv_file, output_variable, output_format ,categories, cons_scale=1,i_a_scale=1):
        self.input_data = pd.read_csv(input_csv_file)
        self.output_data = pd.read_csv(output_csv_file)
        self.output_variable = output_variable 
        #Demand Categories if needed 
        if categories==None and (output_variable=="i_a" or output_variable=='both'):
            print("Category dictionary must be defined for output variable i_a. Training terminated.")
            sys.exit()
        #Subset data inputs
        self.input_data = self.input_data[['Alpha','k','Sigma','Theta']].values
        #Eliminate and Track Duplicate Rows
        u , unique_indices = np.unique(self.input_data, axis=0, return_index=True)

        self.input_data = self.input_data[np.sort(unique_indices)]
        self.output_data = self.output_data.loc[np.sort(unique_indices)]

        
        if output_variable == "both":
            mapping = categories
            self.output_data["Equation"] = self.output_data["Equation"].map(mapping)/i_a_scale
            self.output_data["Consumption"] = self.output_data["Consumption"]/cons_scale
            self.output_data = self.output_data[['Equation','Consumption']].values

        elif output_variable == "i_a":
            if output_format=="decimal":
                mapping=categories
                self.output_data["Equation"] = self.output_data["Equation"].map(mapping)/i_a_scale
                self.output_data = self.output_data[['Equation']].values
            elif output_format=="category":
                categories = list(categories.keys())
                self.output_data.reset_index(drop=True, inplace=True)
                one_hot = np.zeros((len(self.output_data), len(categories)))
                for i, category in enumerate(categories):
                    one_hot[:, i] = (self.output_data["Equation"] == category).astype(int)
                    print( category, sum(one_hot[:,i]))
                self.output_data = one_hot
            else:
                print('Variable output_format must be either "decimal" or "category".')

        elif output_variable == "consumption":
            self.output_data = self.output_data[['Consumption']]/cons_scale.values

        else:
            print(f'Output variable {output_variable} not supported. Try "i_a", "consumption", or "both"')


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):


        input_sample = torch.tensor(self.input_data[idx], dtype=torch.float32)
        output_sample = torch.tensor(self.output_data[idx], dtype=torch.float32)


        return input_sample, output_sample

class BellmanDataLoader(BaseDataLoader):
    """
    Custom class for loading the input/output from the Bellman equation
    """
    def __init__(self, data_dir, batch_size, input_csv_file, output_csv_file, output_variable="both", output_format = None, categories = None, cons_scale=1, i_a_scale=1, shuffle=True, validation_split=0.0, num_workers=1):
        
        self.data_dir = data_dir
        self.dataset = BellmanDataset(input_csv_file, output_csv_file, output_variable, output_format, categories, cons_scale, i_a_scale)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class InformedBellmanDataLoader(BaseDataLoader):
    """
    Custom class for loading the input/output from the Bellman equation and the predicted i_a from a given model.
    """
    def __init__(self, data_dir, batch_size, input_csv_file, output_csv_file, output_variable, output_format, info_model_path, info_model_format, categories=None, shuffle=True, validation_split=0.0, num_workers=1):
        
        self.data_dir = data_dir
        self.dataset = InformedBellmanDataset(input_csv_file, output_csv_file, output_variable, output_format, info_model_path, info_model_format, categories)
    
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)