import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EVAL_FILE = BASE_DIR / "data" / "eval_samples.json"


def load_main_module():
    import importlib.util

    main_path = BASE_DIR / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"main.py not found at: {main_path}")

    spec = importlib.util.spec_from_file_location("benchmark_target_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for main.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_samples():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_FILE}")

    with EVAL_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("data/eval_samples.json must contain a JSON array")

    return data


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def extract_text_from_result(result):
    if isinstance(result, tuple):
        if result:
            return result[0]
        return ""
    return result


def compare_text(expected, actual):
    return normalize_text(expected) == normalize_text(actual)


def run_cleaning_pipeline(module, original_text, language):
    errors = []

    enforce_fluent_output = getattr(module, "enforce_fluent_output", None)
    if callable(enforce_fluent_output):
        try:
            result = enforce_fluent_output(original_text, original_text, language)
            text = extract_text_from_result(result)
            if text is not None:
                return text, "enforce_fluent_output", errors
        except Exception as exc:
            errors.append(f"enforce_fluent_output failed: {exc}")

    clean_stuttered_text_by_language = getattr(module, "clean_stuttered_text_by_language", None)
    if callable(clean_stuttered_text_by_language):
        try:
            result = clean_stuttered_text_by_language(original_text, language)
            text = extract_text_from_result(result)
            if text is not None:
                return text, "clean_stuttered_text_by_language", errors
        except Exception as exc:
            errors.append(f"clean_stuttered_text_by_language failed: {exc}")

    clean_stuttered_text = getattr(module, "clean_stuttered_text", None)
    if callable(clean_stuttered_text):
        try:
            result = clean_stuttered_text(original_text)
            text = extract_text_from_result(result)
            if text is not None:
                return text, "clean_stuttered_text", errors
        except Exception as exc:
            errors.append(f"clean_stuttered_text failed: {exc}")

    details = f" | {'; '.join(errors)}" if errors else ""
    raise RuntimeError(f"No usable cleaning function found in main.py{details}")


def print_sample_result(index, sample, actual, passed, method_name, errors=None):
    language = sample.get("language", "")
    input_text = sample.get("input_text", "")
    expected_text = sample.get("expected_cleaned_text", "")
    notes = sample.get("notes", "")

    print(f"[{index}] Language: {language}")
    print(f"Method:   {method_name}")
    print(f"Input:    {input_text}")
    print(f"Expected: {expected_text}")
    print(f"Actual:   {actual}")
    print(f"Result:   {'PASS' if passed else 'FAIL'}")
    if notes:
        print(f"Notes:    {notes}")
    if errors:
        for error in errors:
            print(f"Warning:  {error}")
    print("-" * 60)


def print_summary(language_stats, overall_passed, overall_total):
    print()
    print("=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)

    for language in sorted(language_stats):
        passed = language_stats[language]["passed"]
        total = language_stats[language]["total"]
        accuracy = (passed / total * 100.0) if total else 0.0
        print(f"{language}: {passed}/{total} ({accuracy:.2f}%)")

    overall_accuracy = (overall_passed / overall_total * 100.0) if overall_total else 0.0
    print("-" * 60)
    print(f"Overall: {overall_passed}/{overall_total} ({overall_accuracy:.2f}%)")


def validate_sample(sample, index):
    required_keys = {"language", "input_text", "expected_cleaned_text", "notes"}
    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise ValueError(f"Sample {index} is missing required keys: {', '.join(missing)}")


def main():
    try:
        module = load_main_module()
        samples = load_samples()
    except Exception as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    language_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    overall_passed = 0
    overall_total = 0

    for index, sample in enumerate(samples, start=1):
        try:
            validate_sample(sample, index)
        except Exception as exc:
            print(f"[{index}] Invalid sample: {exc}")
            print("-" * 60)
            continue

        language = sample.get("language", "")
        input_text = sample.get("input_text", "")
        expected_text = sample.get("expected_cleaned_text", "")

        language_stats[language]["total"] += 1
        overall_total += 1

        try:
            actual_text, method_name, errors = run_cleaning_pipeline(module, input_text, language)
            actual_text = normalize_text(actual_text)
            expected_text_normalized = normalize_text(expected_text)
            passed = compare_text(expected_text_normalized, actual_text)
            if passed:
                language_stats[language]["passed"] += 1
                overall_passed += 1
            print_sample_result(index, sample, actual_text, passed, method_name, errors)
        except Exception as exc:
            print_sample_result(index, sample, f"ERROR: {exc}", False, "none")
            continue

    print_summary(language_stats, overall_passed, overall_total)


if __name__ == "__main__":
    main()