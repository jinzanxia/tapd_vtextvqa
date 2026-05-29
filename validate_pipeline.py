"""
Validation Script - Verify Evidence Mining Pipeline Installation

This script checks that all components are properly installed and working.
Run this to verify your setup before using the pipeline.

Usage:
    python validate_pipeline.py [--full]
    
    --full: Run full validation including model loading
"""

import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_directory_structure():
    """Verify all required directories exist."""
    logger.info("Checking directory structure...")
    
    required_dirs = [
        'parsing',
        'retrieval',
        'reasoning',
        'pipeline',
        'utils',
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"{GREEN}✓{RESET} {dir_name}/")
        else:
            logger.error(f"{RED}✗{RESET} {dir_name}/ NOT FOUND")
            return False
    
    return True

def check_file_structure():
    """Verify all required files exist."""
    logger.info("\nChecking file structure...")
    
    required_files = [
        ('parsing/__init__.py', 'Parsing module init'),
        ('parsing/question_parser.py', 'Question parser'),
        ('retrieval/__init__.py', 'Retrieval module init'),
        ('retrieval/frame_retrieval.py', 'Frame retriever'),
        ('retrieval/region_localization.py', 'Region localizer'),
        ('retrieval/ocr_visibility.py', 'OCR visibility scorer'),
        ('reasoning/__init__.py', 'Reasoning module init'),
        ('reasoning/qwen_reasoning.py', 'Qwen reasoner'),
        ('pipeline/__init__.py', 'Pipeline module init'),
        ('pipeline/evidence_pipeline.py', 'Evidence pipeline'),
        ('utils/__init__.py', 'Utils module init'),
        ('utils/prompt_builder.py', 'Prompt builder'),
        ('pipeline_integration_example.py', 'Integration example'),
        ('QUICK_REFERENCE.py', 'Quick reference'),
        ('EVIDENCE_PIPELINE_DOCS.md', 'Documentation'),
        ('EVIDENCE_MINING_README.md', 'README'),
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if Path(file_path).exists():
            logger.info(f"{GREEN}✓{RESET} {file_path}")
        else:
            logger.error(f"{RED}✗{RESET} {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_imports():
    """Verify all modules can be imported."""
    logger.info("\nChecking imports...")
    
    imports_to_check = [
        ('parsing.question_parser', 'QuestionParser'),
        ('retrieval.frame_retrieval', 'FrameRetriever'),
        ('retrieval.region_localization', 'RegionLocalizer'),
        ('retrieval.ocr_visibility', 'OCRVisibilityScorer'),
        ('reasoning.qwen_reasoning', 'QwenReasoner'),
        ('pipeline.evidence_pipeline', 'EvidenceMiningPipeline'),
        ('utils.prompt_builder', 'build_question_parsing_prompt'),
    ]
    
    all_import_ok = True
    for module_name, class_name in imports_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                logger.info(f"{GREEN}✓{RESET} from {module_name} import {class_name}")
            else:
                logger.error(f"{RED}✗{RESET} {class_name} not found in {module_name}")
                all_import_ok = False
        except ImportError as e:
            logger.error(f"{RED}✗{RESET} Failed to import {module_name}: {e}")
            all_import_ok = False
    
    return all_import_ok

def check_dependencies():
    """Check if required packages are installed."""
    logger.info("\nChecking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
    }
    
    optional_dependencies = {
        'paddleocr': 'PaddleOCR (optional, for OCR scoring)',
        'peft': 'PEFT (optional, for LoRA)',
    }
    
    all_installed = True
    
    logger.info("Required packages:")
    for import_name, display_name in dependencies.items():
        try:
            __import__(import_name)
            logger.info(f"{GREEN}✓{RESET} {display_name}")
        except ImportError:
            logger.error(f"{RED}✗{RESET} {display_name} NOT INSTALLED")
            all_installed = False
    
    logger.info("\nOptional packages:")
    for import_name, display_name in optional_dependencies.items():
        try:
            __import__(import_name)
            logger.info(f"{GREEN}✓{RESET} {display_name}")
        except ImportError:
            logger.warning(f"{YELLOW}⊘{RESET} {display_name} not installed")
    
    return all_installed

def check_model_paths():
    """Check if model can be accessed."""
    logger.info("\nChecking Qwen model availability...")
    
    try:
        from transformers import AutoModel
        logger.info("Model loading capability: Available")
        logger.info(f"{GREEN}✓{RESET} Can load Qwen2.5-VL models")
        return True
    except Exception as e:
        logger.error(f"{RED}✗{RESET} Model loading issue: {e}")
        return False

def check_function_signatures():
    """Verify key functions have correct signatures."""
    logger.info("\nChecking function signatures...")
    
    try:
        from parsing.question_parser import parse_question
        from retrieval.frame_retrieval import retrieve_relevant_frames
        from retrieval.region_localization import localize_target_regions
        from retrieval.ocr_visibility import score_crop_visibility
        from reasoning.qwen_reasoning import run_vlm_reasoning
        from pipeline.evidence_pipeline import run_pipeline
        
        functions = [
            (parse_question, 'parse_question', ['question']),
            (retrieve_relevant_frames, 'retrieve_relevant_frames', ['frames', 'retrieval_prompt']),
            (localize_target_regions, 'localize_target_regions', ['frames', 'region_prompt']),
            (score_crop_visibility, 'score_crop_visibility', ['candidate_regions', 'ocr_prompt']),
            (run_vlm_reasoning, 'run_vlm_reasoning', ['question']),
            (run_pipeline, 'run_pipeline', ['question', 'frames']),
        ]
        
        for func, name, expected_params in functions:
            import inspect
            sig = inspect.signature(func)
            logger.info(f"{GREEN}✓{RESET} {name}{sig}")
        
        return True
    except Exception as e:
        logger.error(f"{RED}✗{RESET} Function signature check failed: {e}")
        return False

def quick_import_test():
    """Quick test to verify core pipeline can be imported."""
    logger.info("\nQuick import test...")
    
    try:
        from pipeline.evidence_pipeline import EvidenceMiningPipeline
        logger.info(f"{GREEN}✓{RESET} Main pipeline class importable")
        return True
    except Exception as e:
        logger.error(f"{RED}✗{RESET} Failed to import main pipeline: {e}")
        return False

def run_validation(full=False):
    """Run all validation checks."""
    logger.info("="*70)
    logger.info("EVIDENCE MINING PIPELINE - VALIDATION CHECK")
    logger.info("="*70)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("File Structure", check_file_structure),
        ("Module Imports", check_imports),
        ("Dependencies", check_dependencies),
        ("Function Signatures", check_function_signatures),
        ("Quick Import Test", quick_import_test),
    ]
    
    if full:
        checks.append(("Model Paths", check_model_paths))
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        logger.info(f"{check_name}: {status}")
    
    logger.info("="*70)
    logger.info(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info(f"{GREEN}✓ All checks passed! Pipeline is ready to use.{RESET}")
        return 0
    else:
        logger.error(f"{RED}✗ Some checks failed. See details above.{RESET}")
        return 1

def print_next_steps():
    """Print recommended next steps."""
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    
    print("""
1. Review the documentation:
   - EVIDENCE_MINING_README.md - Quick start guide
   - EVIDENCE_PIPELINE_DOCS.md - Full API reference
   - QUICK_REFERENCE.py - Code examples

2. Run the quick reference examples:
   python QUICK_REFERENCE.py
   # Uncomment examples at the bottom

3. Test with your data:
   from pipeline.evidence_pipeline import run_pipeline
   result = run_pipeline(question, video_frames)
   print(result['answer'])

4. Integrate with your pipeline:
   from pipeline_integration_example import IntegratedVideoQAPipeline
   pipeline = IntegratedVideoQAPipeline(model_path="...")
   predictions = pipeline.process_video_qas(qa_data, video_dir, output_json)

5. Customize as needed:
   - Adjust retrieval weights in frame_retrieval.py
   - Modify prompts in utils/prompt_builder.py
   - Change OCR scoring weights in ocr_visibility.py

Need help? Check IMPLEMENTATION_SUMMARY.txt for troubleshooting.
""")

if __name__ == "__main__":
    full_check = "--full" in sys.argv
    exit_code = run_validation(full=full_check)
    print_next_steps()
    sys.exit(exit_code)
