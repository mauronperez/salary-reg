import sys
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator

def validate_data(path: str):
    df = pd.read_csv(path)

    # 2. Create an Ephemeral DataContext (no config files needed)
    context = gx.get_context(mode="ephemeral")

    # 3. Wrap DataFrame in Validator via a Batch
    batch = Batch(data=df)
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[batch],
        data_context=context,
    )

    # 4. Define your expectations
    validator.expect_column_values_to_not_be_null("INGRESOS")

    
    validator.expect_column_values_to_be_between("NHIJOBIO", min_value=0)
    validator.expect_column_values_to_be_between("ESTUDIOSA", min_value=0)
    

    # 5. Run validation
    results = validator.validate()
    total = len(results["results"])
    passed = sum(r["success"] for r in results["results"])
    failed = total - passed

    print(f"\n{path}: {passed}/{total} checks passed")
    if failed:
        print("❌ Failed expectations:")
        for r in results["results"]:
            if not r["success"]:
                config = r["expectation_config"]
                column = config.kwargs.get("column", "N/A")
                expectation_type = config.type
                kwargs = {k: v for k, v in config.kwargs.items() if k != "column"}
                print(f"  - {expectation_type} on column '{column}' with params: {kwargs}")
                
                # Show some details about the failure
                result = r.get("result", {})
                if "observed_value" in result:
                    print(f"    Observed: {result['observed_value']}")
                if "element_count" in result and "unexpected_count" in result:
                    print(f"    Unexpected count: {result['unexpected_count']}/{result['element_count']}")
        sys.exit(1)
    else:
        print("✅ All checks passed!")

if __name__ == "__main__":
    for split in ["data/raw/train.csv", "data/raw/eval.csv", "data/raw/holdout.csv"]:
        validate_data(split)