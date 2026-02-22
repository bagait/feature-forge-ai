import pandas as pd
import ollama
import json
import os
import sys

class FeatureForgeAgent:
    """An AI agent for automated feature engineering."""

    def __init__(self, df: pd.DataFrame, model: str = "llama3:8b"):
        """
        Initializes the agent with a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            model (str): The Ollama model to use for generation.
        """
        self.df = df.copy()
        self.model = model
        self.history = []

    def _get_df_summary(self) -> str:
        """Generates a string summary of the DataFrame's structure."""
        summary = "Current DataFrame columns and data types:\n"
        for col, dtype in self.df.dtypes.items():
            summary += f"- {col}: {dtype}\n"
        summary += "\nFirst 3 rows of data:\n"
        summary += self.df.head(3).to_string()
        return summary

    def _get_system_prompt(self) -> str:
        """
        Creates the system prompt for the LLM, guiding its behavior.
        """
        return """
You are an expert data scientist specializing in automated feature engineering. 
Your task is to analyze a pandas DataFrame and propose meaningful new features that could improve the performance of a machine learning model.

Based on the provided DataFrame summary, please suggest exactly 3 new features.

For each feature, provide:
1.  `name`: A short, valid Python variable name for the new column (e.g., 'age_at_signup').
2.  `description`: A brief explanation of why this feature might be useful.
3.  `code`: A single line of Python code using the `df` variable to create the new column. The code should be directly executable. For example: `df['new_feature'] = df['col1'] / df['col2']`

Respond ONLY with a valid JSON object containing a list called "features". Do not include any other text, explanations, or markdown formatting outside of the JSON structure.

Example Response:
{
  "features": [
    {
      "name": "spend_per_item",
      "description": "Calculates the average spending per item, which could indicate user purchasing power or preference for premium products.",
      "code": "df['spend_per_item'] = df['total_spent'] / df['items_purchased']"
    },
    {
      "name": "days_since_signup",
      "description": "Measures user tenure, which can be a strong predictor of loyalty and lifetime value.",
      "code": "df['days_since_signup'] = (pd.to_datetime('today') - pd.to_datetime(df['signup_date'])).dt.days"
    }
  ]
}
"""
    
    def _is_model_available(self):
        """Checks if the specified Ollama model is available locally."""
        try:
            models = ollama.list().get('models', [])
            return any(m['name'].split(':')[0] == self.model.split(':')[0] for m in models)
        except Exception as e:
            print(f"\n[Error] Could not connect to Ollama. Is it running?", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            return False

    def generate_features(self, iterations: int = 1):
        """
        Runs the feature generation loop.

        Args:
            iterations (int): The number of times to ask the LLM for new features.
        """
        if not self._is_model_available():
            print(f"\nModel '{self.model}' not found. Please pull it first with: ollama pull {self.model.split(':')[0]}", file=sys.stderr)
            return self.df
        
        print(f"\nStarting feature generation with model: {self.model}")
        
        client = ollama.Client()
        
        for i in range(iterations):
            print(f"\n--- Iteration {i+1}/{iterations} ---")
            summary = self._get_df_summary()
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": summary}
            ]
            
            try:
                print("Asking LLM to propose new features...")
                response = client.chat(model=self.model, messages=messages, format='json')
                response_content = response['message']['content']
                suggestions = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"[Error] Failed to parse JSON response: {e}")
                print(f"Raw response: {response_content}")
                continue
            except Exception as e:
                print(f"[Error] An unexpected error occurred while communicating with Ollama: {e}")
                continue

            if 'features' not in suggestions or not isinstance(suggestions['features'], list):
                print("[Warning] The response did not contain a 'features' list. Skipping iteration.")
                continue

            for feature in suggestions['features']:
                name = feature.get('name')
                desc = feature.get('description')
                code = feature.get('code')

                if not all([name, desc, code]):
                    print(f"[Warning] Skipping incomplete feature suggestion: {feature}")
                    continue
                
                if name in self.df.columns:
                    print(f"[Info] Feature '{name}' already exists. Skipping.")
                    continue

                print(f"\n\033[1m✨ Generating feature: {name}\033[0m")
                print(f"   \033[90m└─ Rationale: {desc}\033[0m")
                
                try:
                    # Use a local dict for exec to avoid polluting the namespace
                    # Pass the DataFrame via this dict
                    local_scope = {'df': self.df}
                    exec(code, {}, local_scope)
                    self.df = local_scope['df'] # Retrieve the modified DataFrame
                    print(f"   \033[92m✔ Success! Column '{name}' added.\033[0m")
                    self.history.append(feature)
                except Exception as e:
                    print(f"   \033[91m✗ Failed to execute code: {code}\033[0m")
                    print(f"     Error: {e}")
        
        return self.df

def create_sample_data(filename: str = "sample_data.csv"):
    """Creates a sample CSV file for demonstration."""
    if not os.path.exists(filename):
        print(f"Creating sample data file: {filename}")
        data = {
            'user_id': [1, 2, 3, 4, 5],
            'age': [34, 25, 45, 21, 38],
            'signup_date': ['2025-01-15', '2025-06-20', '2024-11-01', '2026-02-01', '2026-01-15'],
            'last_purchase_date': ['2026-02-10', '2026-01-05', '2026-02-20', '2026-02-15', '2025-12-18'],
            'total_spent': [450.50, 120.00, 1500.75, 75.25, 820.00],
            'items_purchased': [12, 3, 25, 2, 18]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Ensure Ollama is running
    print("Welcome to Feature Forge AI! ")
    print("Please ensure the Ollama application is running in the background.")
    
    # Setup sample data
    input_csv = "sample_data.csv"
    output_csv = "augmented_data.csv"
    create_sample_data(input_csv)

    # Load data
    try:
        initial_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[Error] Input file not found: {input_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded data from '{input_csv}'. Initial columns: {initial_df.columns.tolist()}")

    # Initialize and run the agent
    # Using llama3:8b as a good default. Can be changed to mistral, etc.
    agent = FeatureForgeAgent(initial_df, model="llama3:8b")
    augmented_df = agent.generate_features(iterations=2)

    # Save results
    if augmented_df is not None and not augmented_df.equals(initial_df):
        print(f"\nFeature generation complete. Saving augmented data to '{output_csv}'...")
        augmented_df.to_csv(output_csv, index=False)
        print("\nFinal DataFrame head:")
        print(augmented_df.head())
        print(f"\nNew columns: {list(set(augmented_df.columns) - set(initial_df.columns))}")
    else:
        print("\nNo new features were generated.")
