(define (domain blocksworld)
(:requirements :strips :typing :equality :negative-preconditions :conditional-effects)
(:predicates (clear ?x)
             (on-table ?x)
             (on ?x ?y)
             ; MARKS
             (mark ?x) (mark-0 ?x) (mark-1 ?x)
             (some-mark-0) (some-mark-1)
             (markable-0 ?x) (markable-1 ?x)
)

(:action move-b-to-b
  :parameters (?bm ?bf ?bt)
  :precondition (and (clear ?bm) (clear ?bt) (on ?bm ?bf) (not (= ?bm ?bt)) (mark ?bm) (mark ?bt))
  :effect (and (not (clear ?bt)) (not (on ?bm ?bf))
               (on ?bm ?bt) (clear ?bf)))

(:action move-b-to-t
  :parameters (?bm ?bf)
  :precondition (and (clear ?bm) (on ?bm ?bf) (mark ?bm))
  :effect (and (not (on ?bm ?bf))
               (on-table ?bm) (clear ?bf)))

(:action move-t-to-b
  :parameters (?bm ?bt)
  :precondition (and (clear ?bm) (clear ?bt) (on-table ?bm) (not (= ?bm ?bt)) (mark ?bm) (mark ?bt))
  :effect (and (not (clear ?bt)) (not (on-table ?bm))
               (on ?bm ?bt)))

; =====> MARKS
;
; First mark for rank 0 and K > 0
(:action mark-0
  :arguments (?x)
  :precondition (and (markable-0 ?x) (not (mark ?x)) (not (some-mark-0)))
  :effect (and (mark-0 ?x) (not (markable-0 ?x)) (mark ?x) (some-mark-0)
               (forall (?z) (and (markable-1 ?z) (when (mark-1 ?z) (and (not (mark-1 ?z)) (not (mark ?z))))))
               (not (some-mark-1))
               ; Similar for higher ranks
          )
)
(:action mark-1
  :arguments (?x)
  :precondition (and (markable-1 ?x) (not (mark ?z)) (not (some-mark-1)) (some-mark-0))
  :effect (and (mark-1 ?x) (not (markable-1 ?x)) (mark ?x) (some-mark-1)
               ; Similar for higher ranks
          )
)
; Move mark for rank K >= 0
(:action move-mark-0
  :arguments (?x ?y)
  :precondition (and (mark-0 ?x) (markable-0 ?y) (not (mark ?y)))
  :effect (and (not (mark-0 ?x)) (not (mark ?x)) (mark-0 ?y) (not (markable-0 ?y)) (mark ?y)
               (forall (?z) (and (markable-1 ?z) (when (mark-1 ?z) (and (not (mark-1 ?z)) (not (mark?z))))))
               (not (some-mark-1))
               ; Similar for higher ranks
          )
)
(:action move-mark-1
  :arguments (?x ?y)
  :precondition (and (mark-1 ?x) (markable-1 ?y) (not (mark ?y)))
  :effect (and (not (mark-1 ?x)) (not (mark ?x)) (mark-1 ?y) (not (markable-1 ?y)) (mark ?y)
               ; Similar for higher ranks
          )
)

)

