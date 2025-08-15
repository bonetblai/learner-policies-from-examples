

(define (problem BW-rand-10)
(:domain blocksworld-3ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 )
(:init
(on b1 b9)
(on b2 b3)
(on b3 b10)
(on b4 b8)
(on-table b5)
(on-table b6)
(on b7 b1)
(on b8 b6)
(on b9 b4)
(on-table b10)
(clear b2)
(clear b5)
(clear b7)
)
(:goal
(and
(on b1 b5)
(on b2 b7)
(on b4 b2)
(on b5 b8)
(on b6 b1)
(on b7 b6)
(on b8 b3)
(on b9 b10)
(on b10 b4))
)
)


