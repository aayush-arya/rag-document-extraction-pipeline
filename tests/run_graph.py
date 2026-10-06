"""Run the real pipeline (needs GOOGLE_API_KEY):  python -m tests.run_graph [path]"""
import sys

from graph.workflow import create_extraction_graph

file_path = sys.argv[1] if len(sys.argv) > 1 else "data/input/sample.pdf"
result = create_extraction_graph().invoke({"file_path": file_path})

if result.get("error"):
    raise SystemExit(f"Failed: {result['error']}")

data = result["extracted_data"]
print("\n=== LANGGRAPH PIPELINE ===")
print("Elements:", len(result["documents"]), "| chunks:", len(result["chunks"]))
for key, value in data.report.items():
    print(f"  {key}: {value}")
print("\nWarnings:", *data.warnings, sep="\n  - ") if data.warnings else print("No warnings")
print("\nSaved:", result["output_path"], "and", result["json_path"])
