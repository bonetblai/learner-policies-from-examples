import logging, time, os
from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union, Deque
from pathlib import Path

from clingo import Control, Number, Symbol, String, Function
from .returncodes import ClingoExitCode
from ...util import write_file_lines


class _CountDownTimer:
    def __init__(self, seconds: float):
        self._timeout = seconds
        self._unbounded = seconds <= 0
        self._start_time = time.time()

    def reset(self):
        self._start_time = time.time()

    def elapsed_time(self) -> float:
        return time.time() - self._start_time

    def remaining_time(self) -> float:
        return max(0.0, self._timeout - self.elapsed_time())

    def is_expired(self):
        return not self._unbounded and self.remaining_time() == 0

class ASPSolver:
    def __init__(self, arguments: List[str] = None, fact_signatures: List[Tuple[Any]] = None, loads: List[str] = None):
        ctl_arguments = [] if arguments is None else arguments
        self._ctl = Control(arguments=[] if arguments is None else arguments)
        for fact_signature in fact_signatures:
            self._ctl.add(*fact_signature)
        for load in loads:
            self._ctl.load(load)

    @staticmethod
    def make_fact(name: str, *args) -> Tuple[Any]:
        fact_args: List[Any] = []
        for arg in args:
            if type(arg) == str:
                fact_args.append(String(arg))
            elif type(arg) == int:
                fact_args.append(Number(arg))
        return (name, tuple(fact_args))

    def ground(self, *fact_dicts: Tuple[Dict[str, Any]], verbose: bool = False, dump_asp_program: bool = False):
        facts = [fact for fact_dict in fact_dicts for facts in fact_dict.values() for fact in facts]
        logging.info(f"Grounding logic program with {len(facts)} fact(s)...")
        self._ctl.ground(facts + [("base", [])])
        if dump_asp_program:
            logging.info(f"Dumping logic proram to '{Path(os.getcwd()) / 'program.lp'}'...")
            write_file_lines("program.lp", [f"{fact[0]}({','.join([str(arg.number) for arg in fact[1]])}).\n" for fact in facts])
            #for line in [f"{fact[0]}({','.join([str(arg.number) for arg in fact[1]])})." for fact in facts]: print(line)

    def optimize_model(self,
                       max_models: int = 0,
                       timeout_in_seconds_per_step: Optional[float] = None,
                       timeout_in_seconds: Optional[float] = None) -> Tuple[List[Any], List[int], Any]:
        if timeout_in_seconds_per_step is not None and timeout_in_seconds_per_step > 0:
            assert timeout_in_seconds is not None
            return self._optimize_model_with_timeout(max_models=max_models, timeout_in_seconds_per_step=timeout_in_seconds_per_step, timeout_in_seconds=timeout_in_seconds)
        else:
            return self._optimize_model(max_models=max_models)

    def _optimize_model_with_timeout(self,
                                     max_models: int,
                                     timeout_in_seconds_per_step: float,
                                     timeout_in_seconds: float) -> Tuple[List[Any], List[int], Any]:
        """ https://potassco.org/clingo/python-api/current/clingo/solving.html """
        solver_timeout = False
        count_down_timer = _CountDownTimer(timeout_in_seconds)
        try:
            logging.info(f"Bounded optimization with max_models={max_models} and timeouts of {timeout_in_seconds_per_step} and {timeout_in_seconds} second(s) (10x timeout for first model)...")
            with self._ctl.solve(yield_=True, async_=True) as handle:
                last_model = None
                max_models_reached = False
                while True:
                    handle.resume()
                    timeout = min(timeout_in_seconds, timeout_in_seconds_per_step * (10 if last_model is None else 1))
                    wait_status = handle.wait(timeout=timeout)
                    if not wait_status or count_down_timer.is_expired():
                        logging.info(f"    Timeout of {timeout}/{timeout_in_seconds} second(s) reached")
                        if last_model is None:
                            logging.info(f"    [NO MODEL AVAILABLE]")
                        solver_timeout = True
                        break
                    elif not handle.get().satisfiable:
                        logging.info(f"    Handle is not satisfiable")
                        break
                    else:
                        model = handle.model()
                        if model is None:
                            # This only happens when no more models are available; i.e. optimality is proven
                            assert last_model.optimality_proven
                            logging.info(f"    OPTIMUM FOUND")
                            break
                        else:
                            logging.info(f"    Best model: n={model.number}, cost={model.cost}, size={len(list(model.symbols(shown=True)))}, remaining_time={count_down_timer.remaining_time():.02f}")
                            last_model = model
                            if max_models > 0 and last_model.number == max_models:
                                max_models_reached = True
                                break

                #logging.info(f"Hola.0")
                if last_model is not None:
                    assert solver_timeout or max_models_reached or last_model.optimality_proven
                    exit_code = ClingoExitCode.SATISFIABLE if not solver_timeout else ClingoExitCode.TIMEOUT
                    return last_model.symbols(shown=True), last_model.cost, exit_code
                elif solver_timeout:
                    return None, None, ClingoExitCode.TIMEOUT

                #logging.info(f"Hola.1")
                result = handle.get()
                #logging.info(f"Hola.1: result={result}")
                if result.exhausted:
                    return None, None, ClingoExitCode.EXHAUSTED
                elif result.unsatisfiable:
                    return None, None, ClingoExitCode.UNSATISFIABLE
                elif result.unknown:
                    return None, None, ClingoExitCode.UNKNOWN
                elif result.interrupted:
                    return None, None, ClingoExitCode.INTERRUPTED
                else:
                    raise RuntimeError(f"ERROR: Unexpected handle")
        finally:
            #logging.info(f"TRY STATEMENT FINISHING")
            pass
        assert False

    def _optimize_model(self, max_models: int) -> Tuple[List[Any], List[int], Any]:
        """ https://potassco.org/clingo/python-api/current/clingo/solving.html """
        logging.info(f"Unbounded optimization with max_models={max_models}...")
        with self._ctl.solve(yield_=True) as handle:
            last_model = None
            max_models_reached = False
            for model in handle:
                logging.info(f"    Best model: n={model.number}, cost={model.cost}, size={len(list(model.symbols(shown=True)))}")
                last_model = model
                if max_models > 0 and last_model.number == max_models:
                    max_models_reached = True
                    break

            if last_model is not None:
                assert max_models_reached or last_model.optimality_proven
                if not max_models_reached:
                    logging.info(f"    OPTIMUM FOUND")
                return last_model.symbols(shown=True), last_model.cost, ClingoExitCode.SATISFIABLE

            result = handle.get()
            if result.exhausted:
                return None, None, ClingoExitCode.EXHAUSTED
            elif result.unsatisfiable:
                return None, None, ClingoExitCode.UNSATISFIABLE
            elif result.unknown:
                return None, None, ClingoExitCode.UNKNOWN
            elif result.interrupted:
                return None, None, ClingoExitCode.INTERRUPTED

    def first_model(self) -> Tuple[List[Any], Any]:
        """ https://potassco.org/clingo/python-api/current/clingo/solving.html """
        with self._ctl.solve(yield_=True) as handle:
            for model in handle:
                return model.symbols(shown=True), ClingoExitCode.SATISFIABLE
            result = handle.get()
            if result.exhausted:
                return None, ClingoExitCode.EXHAUSTED
            elif result.unsatisfiable:
                return None, ClingoExitCode.UNSATISFIABLE
            elif result.unknown:
                return None, ClingoExitCode.UNKNOWN
            elif result.interrupted:
                return None, ClingoExitCode.INTERRUPTED
