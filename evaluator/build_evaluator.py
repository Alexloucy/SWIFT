from utils.tools import check_availability
from utils.registry import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")

def build_evaluator(cfg, len_query, **kwargs):
    available_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.EVALUATOR, available_evaluators)
    print("Build Evaluator: {}".format(cfg.TEST.EVALUATOR))

    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg, len_query, **kwargs)
