import unittest
import pandas as pd
from unittest.mock import patch
from gex2 import DataInspection 
import random as random
import os

def dynamic_mock_input():
    """Automatically identify and load a CSV file in the current workspace and return continuous column indices."""
    # Get the current working directory
    current_dir = os.getcwd()
    
    # List all files in the current directory and filter for CSV files
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the current directory.")
    
    # Load the first CSV file found
    csv_file_path = os.path.join(current_dir, csv_files[0])
    print(f"Loading CSV file: {csv_file_path}")
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Identify continuous (numeric) columns in the DataFrame
    continuous_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    print("Continuous Columns:", continuous_columns)
    
    # Useful for selecting continuous columns for scatterplot
    if len(continuous_columns) >= 2:
        # Automatically choose the first two continuous columns
        col1 = continuous_columns[0]
        col2 = continuous_columns[1]
        
        # Return the 1-based indices (as strings) for user input simulation
        col1_index = str(df.columns.get_loc(col1) + 1)
        col2_index = str(df.columns.get_loc(col2) + 1)
        
        #return [col1_index, col2_index]
        yield col1_index
        yield col2_index
    else:
        raise ValueError("Not enough continuous columns for a scatterplot.")

class TestDataInspection(unittest.TestCase):

    def setUp(self):
        """Set up a common test environment."""
        self.analysis = DataInspection()
        
        current_dir = os.getcwd()
        
        csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the current directory.")
        
        csv_file_path = os.path.join(current_dir, csv_files[0])
        print(f"Loading CSV file: {csv_file_path}")
        
        self.test_csv = csv_file_path 

        self.analysis.load_csv(self.test_csv)


    # Test Case 1: Load a valid CSV file
    def test_load_csv_valid(self):
        """Test loading a valid CSV file."""
        self.analysis.load_csv(self.test_csv)
        self.assertIsInstance(self.analysis.df, pd.DataFrame)
        self.assertFalse(self.analysis.df.empty) 

    # Test Case 2: Handle missing values in a numeric column with <50% missing
    def test_handle_missing_numeric_column(self):

        
        numeric_column = pd.Series([random.randint(1, 10) for _ in range(100)])

        missing_indices = random.sample(range(100), 5)
        for idx in missing_indices:
            numeric_column[idx] = None
        
        self.analysis.df = pd.DataFrame({'numeric_ordinal': numeric_column})
        
        self.analysis.handle_missing_values('numeric_ordinal')
        
        self.assertFalse(self.analysis.df['numeric_ordinal'].isna().any())  #
    
    # Test Case 3: Handle missing values in a numeric column with >50% missing
    def test_handle_missing_numeric_column_drop(self):
        
        numeric_column = pd.Series([random.randint(1, 10) for _ in range(100)])

        missing_indices = random.sample(range(100), 55)
        for idx in missing_indices:
            numeric_column[idx] = None
        
        self.analysis.df = pd.DataFrame({'numeric_ordinal_more_than_50_percent_missing': numeric_column})
        
        self.analysis.handle_missing_values('numeric_ordinal_more_than_50_percent_missing')
        self.assertNotIn('numeric_ordinal_more_than_50_percent_missing', self.analysis.df.columns)
    
    # Test Case 4: Handle missing values in a non-numeric column
    def test_handle_missing_non_numeric_column(self):
        
        ordinal_values = ['Low', 'Medium', 'High']
        
        non_numeric_column = pd.Series([random.choice(ordinal_values) for _ in range(100)])
        
        missing_indices = random.sample(range(100), 10)
        for idx in missing_indices:
            non_numeric_column[idx] = None
        
        self.analysis.df = pd.DataFrame({'non_numeric_ordinal': non_numeric_column})
        
        self.analysis.handle_missing_values('non_numeric_ordinal')
        
        self.assertFalse(self.analysis.df['non_numeric_ordinal'].isna().any())  # 

    # Test Case 5: Handle incorrect data types
    def test_check_data_types_convert(self):
        
        numeric_looking_strings = pd.Series([str(random.randint(1, 100)) for _ in range(100)])
        
        self.analysis.df = pd.DataFrame({'numeric_ordinal_as_strings': numeric_looking_strings})
        
        self.analysis.check_data_types('numeric_ordinal_as_strings')
        
        self.assertTrue(pd.api.types.is_numeric_dtype(self.analysis.df['numeric_ordinal_as_strings']))
    
    
    # Test Case 6: Calculate median
    @patch('gex2.DataInspection.plot_boxplot')
    def test_calculate_median(self, mock_boxplot):
        
        numeric_ord_column = pd.Series([random.randint(1, 8) for _ in range(100)])
        
        self.analysis.df = pd.DataFrame({'numerical_ordinal': numeric_ord_column})
        
        actual_median = self.analysis.classify_and_calculate('numerical_ordinal')
        expected_median = self.analysis.df['numerical_ordinal'].median()

        self.assertEqual(expected_median, actual_median)  
        mock_boxplot.assert_called_once()
    
    # Test Case 7: Calculate mean
    @patch('gex2.DataInspection.plot_histogram')
    def test_calculate_mean(self, mock_histogram):

        random_floats = [random.uniform(1.0, 100.0) for _ in range(100)]
        
        self.analysis.df = pd.DataFrame({'numeric_continuous': random_floats})
        
        actual_mean_val = self.analysis.classify_and_calculate('numeric_continuous')
        expected_mean_val = self.analysis.df['numeric_continuous'].mean()
        
        self.assertAlmostEqual(expected_mean_val, actual_mean_val, places=1)  
        mock_histogram.assert_called_once()
    
    # Test Case 8: Calculate mode
    @patch('gex2.DataInspection.plot_bar_chart')
    def test_calculate_mode(self, mock_bar_chart):

        nominal_values = ['Red', 'Blue', 'Green']
        
        nominal_columns = pd.Series([random.choice(nominal_values) for _ in range(100)])

        self.analysis.df = pd.DataFrame({'nominal_columns': nominal_columns})

        
        actual_mode_val = self.analysis.classify_and_calculate('nominal_columns')
        expected_mode_val = self.analysis.df['nominal_columns'].mode()[0]
        self.assertEqual(expected_mode_val, actual_mode_val)  
        mock_bar_chart.assert_called_once()

    # Test Case 9: Plotting scatterplot
    @patch('gex2.DataInspection.plot_scatter')
    @patch('builtins.input', side_effect=dynamic_mock_input())
    def test_scatterplot_user_input(self, mock_input, mock_scatterplot):
        
        self.analysis.load_csv(self.test_csv)
        
        self.analysis.ask_for_scatterplot()
        mock_scatterplot.assert_called_once()
    
    
    # Test Case 10: Correlation calculation
    @patch('builtins.input', side_effect=['1', '2'])  
    def test_correlation_user_input(self, mock_input):
                
        numeric_cols = [col for col in self.analysis.df.columns if pd.api.types.is_numeric_dtype(self.analysis.df[col])]
        
        if len(numeric_cols) < 2:
            self.skipTest("Not enough numeric columns in the CSV to perform correlation.")
        
        corr_value = self.analysis.ask_for_correlation(numeric_cols)
        
        col1, col2 = numeric_cols[0], numeric_cols[1] 
        
        expected_corr = self.analysis.df[col1].corr(self.analysis.df[col2])
        
        self.assertAlmostEqual(corr_value, expected_corr)
    
    # Test Case 11: Standard Deviation calculation
    @patch('builtins.input', side_effect=['2']) 
    def test_stddev_user_input(self, mock_input):
        
        
        numeric_cols = [col for col in self.analysis.df.columns if pd.api.types.is_numeric_dtype(self.analysis.df[col])]
        
        if len(numeric_cols) < 2:
            self.skipTest("Not enough numeric columns in the CSV to perform correlation.")
        
        std_value = self.analysis.ask_for_std(numeric_cols)
        
        col1 = numeric_cols[1]  
        
        expected_std = self.analysis.df[col1].std()
        
        self.assertAlmostEqual(std_value, expected_std)
    

    # Test Case 12: Skewness calculation
    @patch('builtins.input', side_effect=['1'])  
    def test_skewness_user_input(self, mock_input):
                
        
        numeric_cols = [col for col in self.analysis.df.columns if pd.api.types.is_numeric_dtype(self.analysis.df[col])]
                
        skew_value = self.analysis.ask_for_skewness(numeric_cols)
        
        col1 = numeric_cols[0]  
        
        expected_skew = self.analysis.df[col1].skew()
        
        self.assertAlmostEqual(skew_value, expected_skew)
    


    # Test Case 13: Kurtosis calculation
    @patch('builtins.input', side_effect=['2'])  
    def test_kurtosis_user_input(self, mock_input):
                
        
        numeric_cols = [col for col in self.analysis.df.columns if pd.api.types.is_numeric_dtype(self.analysis.df[col])]
        
        kurt_value = self.analysis.ask_for_kurtosis(numeric_cols)
        
        col1 = numeric_cols[1]  
        
        expected_kurt = self.analysis.df[col1].kurt()
        
        self.assertAlmostEqual(kurt_value, expected_kurt)


if __name__ == '__main__':
    unittest.main()
