(define (problem ZTRAVEL-3-1)
(:domain zeno-travel)
(:objects
	plane1
	plane2
	plane3
	person1
	city0
	city1
	city2
	city3
	fl0
	fl1
	fl2
	fl3
	fl4
	fl5
	fl6
	)
(:init
	(at plane1 city2)
	(aircraft plane1)
	(fuel-level plane1 fl0)
	(at plane2 city0)
	(aircraft plane2)
	(fuel-level plane2 fl0)
	(at plane3 city2)
	(aircraft plane3)
	(fuel-level plane3 fl0)
	(at person1 city0)
	(person person1)
	(city city0)
	(city city1)
	(city city2)
	(city city3)
	(next fl0 fl1)
	(next fl1 fl2)
	(next fl2 fl3)
	(next fl3 fl4)
	(next fl4 fl5)
	(next fl5 fl6)
	(flevel fl0)
	(flevel fl1)
	(flevel fl2)
	(flevel fl3)
	(flevel fl4)
	(flevel fl5)
	(flevel fl6)
)
(:goal (and
	(at plane1 city2)
	(at person1 city3)
	))

)
