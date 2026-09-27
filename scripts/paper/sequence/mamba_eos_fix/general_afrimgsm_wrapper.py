"""Run only the AfriMGSM tasks of the frozen General prompt runner (same task names, prompts, batch settings)."""
import sys
import run_general_prompt_lm_eval as runner

def afrimgsm_only(selection):
    return {"closed_and_generation": [f"afrimgsm_{language}_prompt_1" for language in runner.LANGUAGES]}

runner.test_task_groups = afrimgsm_only
runner.main()
