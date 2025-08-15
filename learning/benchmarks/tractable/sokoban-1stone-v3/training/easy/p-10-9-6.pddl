;;  1234567890
;; 1##########
;; 2#       @#
;; 3#     .  #
;; 4#        #
;; 5#        #
;; 6#        #
;; 7#   $    #
;; 8#        #
;; 9##########

(define (problem x-microban-sequential-1stone-10-9-42-73-56)
  (:domain sokoban-sequential)
  (:objects
      pos-1-1 - location
      pos-1-2 - location
      pos-1-3 - location
      pos-1-4 - location
      pos-1-5 - location
      pos-1-6 - location
      pos-1-7 - location
      pos-1-8 - location
      pos-1-9 - location
      pos-2-1 - location
      pos-2-2 - location
      pos-2-3 - location
      pos-2-4 - location
      pos-2-5 - location
      pos-2-6 - location
      pos-2-7 - location
      pos-2-8 - location
      pos-2-9 - location
      pos-3-1 - location
      pos-3-2 - location
      pos-3-3 - location
      pos-3-4 - location
      pos-3-5 - location
      pos-3-6 - location
      pos-3-7 - location
      pos-3-8 - location
      pos-3-9 - location
      pos-4-1 - location
      pos-4-2 - location
      pos-4-3 - location
      pos-4-4 - location
      pos-4-5 - location
      pos-4-6 - location
      pos-4-7 - location
      pos-4-8 - location
      pos-4-9 - location
      pos-5-1 - location
      pos-5-2 - location
      pos-5-3 - location
      pos-5-4 - location
      pos-5-5 - location
      pos-5-6 - location
      pos-5-7 - location
      pos-5-8 - location
      pos-5-9 - location
      pos-6-1 - location
      pos-6-2 - location
      pos-6-3 - location
      pos-6-4 - location
      pos-6-5 - location
      pos-6-6 - location
      pos-6-7 - location
      pos-6-8 - location
      pos-6-9 - location
      pos-7-1 - location
      pos-7-2 - location
      pos-7-3 - location
      pos-7-4 - location
      pos-7-5 - location
      pos-7-6 - location
      pos-7-7 - location
      pos-7-8 - location
      pos-7-9 - location
      pos-8-1 - location
      pos-8-2 - location
      pos-8-3 - location
      pos-8-4 - location
      pos-8-5 - location
      pos-8-6 - location
      pos-8-7 - location
      pos-8-8 - location
      pos-8-9 - location
      pos-9-1 - location
      pos-9-2 - location
      pos-9-3 - location
      pos-9-4 - location
      pos-9-5 - location
      pos-9-6 - location
      pos-9-7 - location
      pos-9-8 - location
      pos-9-9 - location
      pos-10-1 - location
      pos-10-2 - location
      pos-10-3 - location
      pos-10-4 - location
      pos-10-5 - location
      pos-10-6 - location
      pos-10-7 - location
      pos-10-8 - location
      pos-10-9 - location
      stone-01 - stone
  )
  (:init
      (IS-NONGOAL pos-1-1)
      (IS-NONGOAL pos-1-2)
      (IS-NONGOAL pos-1-3)
      (IS-NONGOAL pos-1-4)
      (IS-NONGOAL pos-1-5)
      (IS-NONGOAL pos-1-6)
      (IS-NONGOAL pos-1-7)
      (IS-NONGOAL pos-1-8)
      (IS-NONGOAL pos-1-9)
      (IS-NONGOAL pos-2-1)
      (IS-NONGOAL pos-2-2)
      (IS-NONGOAL pos-2-3)
      (IS-NONGOAL pos-2-4)
      (IS-NONGOAL pos-2-5)
      (IS-NONGOAL pos-2-6)
      (IS-NONGOAL pos-2-7)
      (IS-NONGOAL pos-2-8)
      (IS-NONGOAL pos-2-9)
      (IS-NONGOAL pos-3-1)
      (IS-NONGOAL pos-3-2)
      (IS-NONGOAL pos-3-3)
      (IS-NONGOAL pos-3-4)
      (IS-NONGOAL pos-3-5)
      (IS-NONGOAL pos-3-6)
      (IS-NONGOAL pos-3-7)
      (IS-NONGOAL pos-3-8)
      (IS-NONGOAL pos-3-9)
      (IS-NONGOAL pos-4-1)
      (IS-NONGOAL pos-4-2)
      (IS-NONGOAL pos-4-3)
      (IS-NONGOAL pos-4-4)
      (IS-NONGOAL pos-4-5)
      (IS-NONGOAL pos-4-6)
      (IS-NONGOAL pos-4-7)
      (IS-NONGOAL pos-4-8)
      (IS-NONGOAL pos-4-9)
      (IS-NONGOAL pos-5-1)
      (IS-NONGOAL pos-5-2)
      (IS-NONGOAL pos-5-3)
      (IS-NONGOAL pos-5-4)
      (IS-NONGOAL pos-5-5)
      (IS-NONGOAL pos-5-6)
      (IS-NONGOAL pos-5-7)
      (IS-NONGOAL pos-5-8)
      (IS-NONGOAL pos-5-9)
      (IS-NONGOAL pos-6-1)
      (IS-NONGOAL pos-6-2)
      (IS-NONGOAL pos-6-3)
      (IS-NONGOAL pos-6-4)
      (IS-NONGOAL pos-6-5)
      (IS-NONGOAL pos-6-6)
      (IS-NONGOAL pos-6-7)
      (IS-NONGOAL pos-6-8)
      (IS-NONGOAL pos-6-9)
      (IS-NONGOAL pos-7-1)
      (IS-NONGOAL pos-7-2)
      (IS-GOAL pos-7-3)
      (IS-NONGOAL pos-7-4)
      (IS-NONGOAL pos-7-5)
      (IS-NONGOAL pos-7-6)
      (IS-NONGOAL pos-7-7)
      (IS-NONGOAL pos-7-8)
      (IS-NONGOAL pos-7-9)
      (IS-NONGOAL pos-8-1)
      (IS-NONGOAL pos-8-2)
      (IS-NONGOAL pos-8-3)
      (IS-NONGOAL pos-8-4)
      (IS-NONGOAL pos-8-5)
      (IS-NONGOAL pos-8-6)
      (IS-NONGOAL pos-8-7)
      (IS-NONGOAL pos-8-8)
      (IS-NONGOAL pos-8-9)
      (IS-NONGOAL pos-9-1)
      (IS-NONGOAL pos-9-2)
      (IS-NONGOAL pos-9-3)
      (IS-NONGOAL pos-9-4)
      (IS-NONGOAL pos-9-5)
      (IS-NONGOAL pos-9-6)
      (IS-NONGOAL pos-9-7)
      (IS-NONGOAL pos-9-8)
      (IS-NONGOAL pos-9-9)
      (IS-NONGOAL pos-10-1)
      (IS-NONGOAL pos-10-2)
      (IS-NONGOAL pos-10-3)
      (IS-NONGOAL pos-10-4)
      (IS-NONGOAL pos-10-5)
      (IS-NONGOAL pos-10-6)
      (IS-NONGOAL pos-10-7)
      (IS-NONGOAL pos-10-8)
      (IS-NONGOAL pos-10-9)

      (ADJ-RIGHT pos-1-1 pos-2-1)
      (ADJ-RIGHT pos-1-1 pos-1-2)
      (ADJ-RIGHT pos-1-2 pos-2-2)
      (ADJ-RIGHT pos-1-2 pos-1-3)
      (ADJ-RIGHT pos-1-3 pos-2-3)
      (ADJ-RIGHT pos-1-3 pos-1-4)
      (ADJ-RIGHT pos-1-4 pos-2-4)
      (ADJ-RIGHT pos-1-4 pos-1-5)
      (ADJ-RIGHT pos-1-5 pos-2-5)
      (ADJ-RIGHT pos-1-5 pos-1-6)
      (ADJ-RIGHT pos-1-6 pos-2-6)
      (ADJ-RIGHT pos-1-6 pos-1-7)
      (ADJ-RIGHT pos-1-7 pos-2-7)
      (ADJ-RIGHT pos-1-7 pos-1-8)
      (ADJ-RIGHT pos-1-8 pos-2-8)
      (ADJ-RIGHT pos-1-8 pos-1-9)
      (ADJ-RIGHT pos-1-9 pos-2-9)
      (ADJ-RIGHT pos-2-1 pos-3-1)
      (ADJ-RIGHT pos-2-1 pos-2-2)
      (ADJ-RIGHT pos-2-2 pos-3-2)
      (ADJ-RIGHT pos-2-2 pos-2-3)
      (ADJ-RIGHT pos-2-3 pos-3-3)
      (ADJ-RIGHT pos-2-3 pos-2-4)
      (ADJ-RIGHT pos-2-4 pos-3-4)
      (ADJ-RIGHT pos-2-4 pos-2-5)
      (ADJ-RIGHT pos-2-5 pos-3-5)
      (ADJ-RIGHT pos-2-5 pos-2-6)
      (ADJ-RIGHT pos-2-6 pos-3-6)
      (ADJ-RIGHT pos-2-6 pos-2-7)
      (ADJ-RIGHT pos-2-7 pos-3-7)
      (ADJ-RIGHT pos-2-7 pos-2-8)
      (ADJ-RIGHT pos-2-8 pos-3-8)
      (ADJ-RIGHT pos-2-8 pos-2-9)
      (ADJ-RIGHT pos-2-9 pos-3-9)
      (ADJ-RIGHT pos-3-1 pos-4-1)
      (ADJ-RIGHT pos-3-1 pos-3-2)
      (ADJ-RIGHT pos-3-2 pos-4-2)
      (ADJ-RIGHT pos-3-2 pos-3-3)
      (ADJ-RIGHT pos-3-3 pos-4-3)
      (ADJ-RIGHT pos-3-3 pos-3-4)
      (ADJ-RIGHT pos-3-4 pos-4-4)
      (ADJ-RIGHT pos-3-4 pos-3-5)
      (ADJ-RIGHT pos-3-5 pos-4-5)
      (ADJ-RIGHT pos-3-5 pos-3-6)
      (ADJ-RIGHT pos-3-6 pos-4-6)
      (ADJ-RIGHT pos-3-6 pos-3-7)
      (ADJ-RIGHT pos-3-7 pos-4-7)
      (ADJ-RIGHT pos-3-7 pos-3-8)
      (ADJ-RIGHT pos-3-8 pos-4-8)
      (ADJ-RIGHT pos-3-8 pos-3-9)
      (ADJ-RIGHT pos-3-9 pos-4-9)
      (ADJ-RIGHT pos-4-1 pos-5-1)
      (ADJ-RIGHT pos-4-1 pos-4-2)
      (ADJ-RIGHT pos-4-2 pos-5-2)
      (ADJ-RIGHT pos-4-2 pos-4-3)
      (ADJ-RIGHT pos-4-3 pos-5-3)
      (ADJ-RIGHT pos-4-3 pos-4-4)
      (ADJ-RIGHT pos-4-4 pos-5-4)
      (ADJ-RIGHT pos-4-4 pos-4-5)
      (ADJ-RIGHT pos-4-5 pos-5-5)
      (ADJ-RIGHT pos-4-5 pos-4-6)
      (ADJ-RIGHT pos-4-6 pos-5-6)
      (ADJ-RIGHT pos-4-6 pos-4-7)
      (ADJ-RIGHT pos-4-7 pos-5-7)
      (ADJ-RIGHT pos-4-7 pos-4-8)
      (ADJ-RIGHT pos-4-8 pos-5-8)
      (ADJ-RIGHT pos-4-8 pos-4-9)
      (ADJ-RIGHT pos-4-9 pos-5-9)
      (ADJ-RIGHT pos-5-1 pos-6-1)
      (ADJ-RIGHT pos-5-1 pos-5-2)
      (ADJ-RIGHT pos-5-2 pos-6-2)
      (ADJ-RIGHT pos-5-2 pos-5-3)
      (ADJ-RIGHT pos-5-3 pos-6-3)
      (ADJ-RIGHT pos-5-3 pos-5-4)
      (ADJ-RIGHT pos-5-4 pos-6-4)
      (ADJ-RIGHT pos-5-4 pos-5-5)
      (ADJ-RIGHT pos-5-5 pos-6-5)
      (ADJ-RIGHT pos-5-5 pos-5-6)
      (ADJ-RIGHT pos-5-6 pos-6-6)
      (ADJ-RIGHT pos-5-6 pos-5-7)
      (ADJ-RIGHT pos-5-7 pos-6-7)
      (ADJ-RIGHT pos-5-7 pos-5-8)
      (ADJ-RIGHT pos-5-8 pos-6-8)
      (ADJ-RIGHT pos-5-8 pos-5-9)
      (ADJ-RIGHT pos-5-9 pos-6-9)
      (ADJ-RIGHT pos-6-1 pos-7-1)
      (ADJ-RIGHT pos-6-1 pos-6-2)
      (ADJ-RIGHT pos-6-2 pos-7-2)
      (ADJ-RIGHT pos-6-2 pos-6-3)
      (ADJ-RIGHT pos-6-3 pos-7-3)
      (ADJ-RIGHT pos-6-3 pos-6-4)
      (ADJ-RIGHT pos-6-4 pos-7-4)
      (ADJ-RIGHT pos-6-4 pos-6-5)
      (ADJ-RIGHT pos-6-5 pos-7-5)
      (ADJ-RIGHT pos-6-5 pos-6-6)
      (ADJ-RIGHT pos-6-6 pos-7-6)
      (ADJ-RIGHT pos-6-6 pos-6-7)
      (ADJ-RIGHT pos-6-7 pos-7-7)
      (ADJ-RIGHT pos-6-7 pos-6-8)
      (ADJ-RIGHT pos-6-8 pos-7-8)
      (ADJ-RIGHT pos-6-8 pos-6-9)
      (ADJ-RIGHT pos-6-9 pos-7-9)
      (ADJ-RIGHT pos-7-1 pos-8-1)
      (ADJ-RIGHT pos-7-1 pos-7-2)
      (ADJ-RIGHT pos-7-2 pos-8-2)
      (ADJ-RIGHT pos-7-2 pos-7-3)
      (ADJ-RIGHT pos-7-3 pos-8-3)
      (ADJ-RIGHT pos-7-3 pos-7-4)
      (ADJ-RIGHT pos-7-4 pos-8-4)
      (ADJ-RIGHT pos-7-4 pos-7-5)
      (ADJ-RIGHT pos-7-5 pos-8-5)
      (ADJ-RIGHT pos-7-5 pos-7-6)
      (ADJ-RIGHT pos-7-6 pos-8-6)
      (ADJ-RIGHT pos-7-6 pos-7-7)
      (ADJ-RIGHT pos-7-7 pos-8-7)
      (ADJ-RIGHT pos-7-7 pos-7-8)
      (ADJ-RIGHT pos-7-8 pos-8-8)
      (ADJ-RIGHT pos-7-8 pos-7-9)
      (ADJ-RIGHT pos-7-9 pos-8-9)
      (ADJ-RIGHT pos-8-1 pos-9-1)
      (ADJ-RIGHT pos-8-1 pos-8-2)
      (ADJ-RIGHT pos-8-2 pos-9-2)
      (ADJ-RIGHT pos-8-2 pos-8-3)
      (ADJ-RIGHT pos-8-3 pos-9-3)
      (ADJ-RIGHT pos-8-3 pos-8-4)
      (ADJ-RIGHT pos-8-4 pos-9-4)
      (ADJ-RIGHT pos-8-4 pos-8-5)
      (ADJ-RIGHT pos-8-5 pos-9-5)
      (ADJ-RIGHT pos-8-5 pos-8-6)
      (ADJ-RIGHT pos-8-6 pos-9-6)
      (ADJ-RIGHT pos-8-6 pos-8-7)
      (ADJ-RIGHT pos-8-7 pos-9-7)
      (ADJ-RIGHT pos-8-7 pos-8-8)
      (ADJ-RIGHT pos-8-8 pos-9-8)
      (ADJ-RIGHT pos-8-8 pos-8-9)
      (ADJ-RIGHT pos-8-9 pos-9-9)
      (ADJ-RIGHT pos-9-1 pos-10-1)
      (ADJ-RIGHT pos-9-1 pos-9-2)
      (ADJ-RIGHT pos-9-2 pos-10-2)
      (ADJ-RIGHT pos-9-2 pos-9-3)
      (ADJ-RIGHT pos-9-3 pos-10-3)
      (ADJ-RIGHT pos-9-3 pos-9-4)
      (ADJ-RIGHT pos-9-4 pos-10-4)
      (ADJ-RIGHT pos-9-4 pos-9-5)
      (ADJ-RIGHT pos-9-5 pos-10-5)
      (ADJ-RIGHT pos-9-5 pos-9-6)
      (ADJ-RIGHT pos-9-6 pos-10-6)
      (ADJ-RIGHT pos-9-6 pos-9-7)
      (ADJ-RIGHT pos-9-7 pos-10-7)
      (ADJ-RIGHT pos-9-7 pos-9-8)
      (ADJ-RIGHT pos-9-8 pos-10-8)
      (ADJ-RIGHT pos-9-8 pos-9-9)
      (ADJ-RIGHT pos-9-9 pos-10-9)
      (ADJ-RIGHT pos-10-1 pos-10-2)
      (ADJ-RIGHT pos-10-2 pos-10-3)
      (ADJ-RIGHT pos-10-3 pos-10-4)
      (ADJ-RIGHT pos-10-4 pos-10-5)
      (ADJ-RIGHT pos-10-5 pos-10-6)
      (ADJ-RIGHT pos-10-6 pos-10-7)
      (ADJ-RIGHT pos-10-7 pos-10-8)
      (ADJ-RIGHT pos-10-8 pos-10-9)

      (player pos-9-2)
      (at stone-01 pos-5-7)
      (clear pos-2-2)
      (clear pos-2-3)
      (clear pos-2-4)
      (clear pos-2-5)
      (clear pos-2-6)
      (clear pos-2-7)
      (clear pos-2-8)
      (clear pos-3-2)
      (clear pos-3-3)
      (clear pos-3-4)
      (clear pos-3-5)
      (clear pos-3-6)
      (clear pos-3-7)
      (clear pos-3-8)
      (clear pos-4-2)
      (clear pos-4-3)
      (clear pos-4-4)
      (clear pos-4-5)
      (clear pos-4-6)
      (clear pos-4-7)
      (clear pos-4-8)
      (clear pos-5-2)
      (clear pos-5-3)
      (clear pos-5-4)
      (clear pos-5-5)
      (clear pos-5-6)
      (clear pos-5-8)
      (clear pos-6-2)
      (clear pos-6-3)
      (clear pos-6-4)
      (clear pos-6-5)
      (clear pos-6-6)
      (clear pos-6-7)
      (clear pos-6-8)
      (clear pos-7-2)
      (clear pos-7-3)
      (clear pos-7-4)
      (clear pos-7-5)
      (clear pos-7-6)
      (clear pos-7-7)
      (clear pos-7-8)
      (clear pos-8-2)
      (clear pos-8-3)
      (clear pos-8-4)
      (clear pos-8-5)
      (clear pos-8-6)
      (clear pos-8-7)
      (clear pos-8-8)
      (clear pos-9-3)
      (clear pos-9-4)
      (clear pos-9-5)
      (clear pos-9-6)
      (clear pos-9-7)
      (clear pos-9-8)
  )
  (:goal (at-goal stone-01))
)
