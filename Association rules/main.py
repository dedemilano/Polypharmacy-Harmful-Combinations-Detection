import os
import hydra
from omegaconf import DictConfig
import mlflow
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from typing import List, Dict, Any

class PolypharmacyAssociationRules:
    def __init__(self, data_path: str):
        """
        Initialize the association rules analysis for polypharmacy data
        
        Args:
            data_path (str): Path to the polypharmacy dataset
        """
        self.data = pd.read_csv(data_path)
        self.hospitalized_data = None
        self.non_hospitalized_data = None
        
    def preprocess_data(self):
        """
        Divide data into hospitalized and non-hospitalized groups
        """
        # Convert drug columns to boolean
        drug_columns = [col for col in self.data.columns if col.startswith('Drug_')]
        self.data[drug_columns] = self.data[drug_columns].astype(bool)
        
        # Split data based on hospitalization
        self.hospitalized_data = self.data[self.data['hospit'] == 1]
        self.non_hospitalized_data = self.data[self.data['hospit'] == 0]
        
    def find_frequent_itemsets(
        self, 
        data: pd.DataFrame, 
        min_support: float = 0.03, 
        max_len: int = 14
    ) -> pd.DataFrame:
        """
        Find frequent itemsets using Apriori algorithm
        
        Args:
            data (pd.DataFrame): Input dataframe
            min_support (float): Minimum support threshold
            max_len (int): Maximum length of itemsets
        
        Returns:
            pd.DataFrame: Frequent itemsets
        """
        drug_columns = [col for col in data.columns if col.startswith('Drug_')]
        
        # Create binary transaction dataframe
        transactions = data[drug_columns]
        
        # Find frequent itemsets
        frequent_itemsets = apriori(
            transactions, 
            min_support=min_support, 
            max_len=max_len, 
            use_colnames=True
        )
        
        return frequent_itemsets
    
    def validate_frequent_itemsets(
        self, 
        frequent_itemsets: pd.DataFrame, 
        min_batch_occurrence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Validate frequent itemsets across multiple data batches
        
        Args:
            frequent_itemsets (pd.DataFrame): Frequent itemsets
            min_batch_occurrence (float): Minimum occurrence across batches
        
        Returns:
            List of validated frequent itemsets
        """
        drug_columns = [col for col in self.hospitalized_data.columns if col.startswith('Drug_')]
        batch_size = 10000
        
        validated_itemsets = []
        
        # Divide hospitalized data into batches
        for batch_start in range(0, len(self.hospitalized_data), batch_size):
            batch_data = self.hospitalized_data.iloc[batch_start:batch_start+batch_size]
            
            batch_frequent_itemsets = self.find_frequent_itemsets(batch_data)
            
            for _, itemset in frequent_itemsets.iterrows():
                # Check if itemset is frequent in this batch
                itemset_match = batch_frequent_itemsets[
                    batch_frequent_itemsets['itemsets'] == itemset['itemsets']
                ]
                
                # Validate against non-hospitalized data
                if not itemset_match.empty:
                    non_hosp_support = self.check_non_hospitalized_support(itemset['itemsets'])
                    
                    validated_itemsets.append({
                        'itemset': itemset['itemsets'],
                        'support': itemset['support'],
                        'non_hospitalized_support': non_hosp_support
                    })
        
        return validated_itemsets
    
    def check_non_hospitalized_support(self, itemset):
        """
        Check support of itemset in non-hospitalized data
        
        Args:
            itemset (frozenset): Set of drugs to check
        
        Returns:
            float: Support in non-hospitalized data
        """
        drug_columns = [col for col in self.non_hospitalized_data.columns if col.startswith('Drug_')]
        
        # Check if all drugs in itemset are present
        mask = np.all(
            self.non_hospitalized_data[list(itemset)] == True, 
            axis=1
        )
        
        return mask.mean()
    
    @hydra.main(config_path="config", config_name="association_rules")
    def run_analysis(self, cfg: DictConfig):
        """
        Main analysis method with MLflow tracking
        
        Args:
            cfg (DictConfig): Configuration from Hydra
        """
        mlflow.set_experiment("polypharmacy_association_rules")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "min_support": cfg.min_support,
                "max_itemset_length": cfg.max_itemset_length,
                "min_batch_occurrence": cfg.min_batch_occurrence
            })
            
            # Preprocess data
            self.preprocess_data()
            
            # Find frequent itemsets in hospitalized data
            frequent_itemsets = self.find_frequent_itemsets(
                self.hospitalized_data, 
                min_support=cfg.min_support,
                max_len=cfg.max_itemset_length
            )
            
            # Validate itemsets
            validated_itemsets = self.validate_frequent_itemsets(
                frequent_itemsets, 
                min_batch_occurrence=cfg.min_batch_occurrence
            )
            
            # Log metrics and artifacts
            mlflow.log_metric("validated_itemsets_count", len(validated_itemsets))
            
            # Save validated itemsets
            validated_df = pd.DataFrame(validated_itemsets)
            validated_df.to_csv("validated_drug_combinations.csv", index=False)
            mlflow.log_artifact("validated_drug_combinations.csv")
            
            return validated_itemsets

def main():
    data_path = "polypharmacy_dataset.csv"
    analyzer = PolypharmacyAssociationRules(data_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()