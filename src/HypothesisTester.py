import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

class ABTest:
    def __init__(self, df):
        self.df = df.copy()  # Create a copy to avoid modifying the original DataFrame
        self.alpha = 0.05  # Significance level

    def perform_test(self, group_col, value_col, test_type="chi2"):
        """Performs A/B testing.

        Args:
            group_col: Column name for grouping (e.g., 'Province', 'PostalCode', 'Gender').
            value_col: Column name for the metric (e.g., 'TotalClaims' for risk, 'TotalPremium' for margin).
            test_type: "chi2" for categorical data, "ttest" for numerical data.

        Returns:
            A dictionary containing the test results (p-value, etc.), or None if an error occurs.
        """

        try:
            if test_type == "chi2":
                contingency_table = pd.crosstab(self.df[group_col], self.df[value_col])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                return {"p_value": p, "chi2": chi2, "dof": dof, "expected": expected}
            elif test_type == "ttest":
                groups = self.df[group_col].unique()
                if len(groups) != 2:
                  print(f"Column {group_col} has {len(groups)} unique values, ttest only works with 2 groups.")
                  return None
                group1 = self.df[self.df[group_col] == groups[0]][value_col].dropna()
                group2 = self.df[self.df[group_col] == groups[1]][value_col].dropna()
                if len(group1) == 0 or len(group2) == 0:
                    print(f"One of the groups for {group_col} has no data after dropping NaN values.")
                    return None
                t_statistic, p_value = ttest_ind(group1, group2)
                return {"p_value": p_value, "t_statistic": t_statistic}
            else:
                print("Invalid test type. Choose 'chi2' or 'ttest'.")
                return None
        except ValueError as e:
            print(f"Error during testing: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


    def analyze_and_report(self, test_results, hypothesis_text):
        """Analyzes test results and prints a report.

        Args:
            test_results: The dictionary returned by perform_test().
            hypothesis_text: Text describing the null hypothesis.
        """
        if test_results is None:
            return
        p_value = test_results.get("p_value")
        if p_value is None:
            print("P-value not found in test results.")
            return

        print(f"Null Hypothesis: {hypothesis_text}")
        print(f"P-value: {p_value}")

        if p_value < self.alpha:
            print("Reject the null hypothesis. There is a statistically significant difference.")
        else:
            print("Fail to reject the null hypothesis. There is no statistically significant difference.")
        print("-" * 50)


# # Example usage (assuming you have your DataFrame loaded as 'df'):
# # Sample DataFrame creation (replace with your actual data)
# np.random.seed(42)
# data = {'Province': np.random.choice(['A', 'B', 'C'], 100),
#         'PostalCode': np.random.choice([1000, 2000, 3000], 100),
#         'Gender': np.random.choice(['Male', 'Female'], 100),
#         'TotalClaims': np.random.randint(0, 1000, 100),
#         'TotalPremium': np.random.randint(100, 500, 100)}
# df = pd.DataFrame(data)

# ab_test = ABTest(df)

# # Test 1: Risk differences across provinces (using TotalClaims as risk proxy)
# results_provinces = ab_test.perform_test("Province", "TotalClaims")
# ab_test.analyze_and_report(results_provinces, "There are no risk differences across provinces")

# # Test 2: Risk differences between zip codes (using TotalClaims as risk proxy)
# results_zipcodes_risk = ab_test.perform_test("PostalCode", "TotalClaims")
# ab_test.analyze_and_report(results_zipcodes_risk, "There are no risk differences between zip codes")

# # Test 3: Profit difference between zip codes (using TotalPremium as profit proxy)
# results_zipcodes_profit = ab_test.perform_test("PostalCode", "TotalPremium", test_type="ttest")
# ab_test.analyze_and_report(results_zipcodes_profit, "There are no significant margin (profit) difference between zip codes")

# # Test 4: Risk difference between Women and Men (using TotalClaims as risk proxy)
# results_gender = ab_test.perform_test("Gender", "TotalClaims", test_type="ttest")
# ab_test.analyze_and_report(results_gender, "There is no significant risk difference between Women and Men")