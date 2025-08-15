(define (problem ZTRAVEL-1-12)
(:domain zeno-travel)
(:objects
	plane1
	person1
	person2
	person3
	person4
	person5
	person6
	person7
	person8
	person9
	person10
	person11
	person12
	city0
	city1
	city2
	city3
	city4
	city5
	city6
	city7
	city8
	city9
	city10
	city11
	city12
	city13
	fl0
	fl1
	fl2
	fl3
	fl4
	fl5
	fl6
	)
(:init
	(at plane1 city4)
	(aircraft plane1)
	(fuel-level plane1 fl0)
	(at person1 city13)
	(person person1)
	(at person2 city6)
	(person person2)
	(at person3 city7)
	(person person3)
	(at person4 city7)
	(person person4)
	(at person5 city1)
	(person person5)
	(at person6 city5)
	(person person6)
	(at person7 city3)
	(person person7)
	(at person8 city12)
	(person person8)
	(at person9 city8)
	(person person9)
	(at person10 city10)
	(person person10)
	(at person11 city6)
	(person person11)
	(at person12 city2)
	(person person12)
	(city city0)
	(city city1)
	(city city2)
	(city city3)
	(city city4)
	(city city5)
	(city city6)
	(city city7)
	(city city8)
	(city city9)
	(city city10)
	(city city11)
	(city city12)
	(city city13)
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
	(at plane1 city9)
	(at person1 city3)
	(at person2 city12)
	(at person3 city8)
	(at person4 city6)
	(at person5 city9)
	(at person6 city4)
	(at person7 city12)
	(at person8 city4)
	(at person9 city2)
	(at person10 city7)
	(at person11 city6)
	(at person12 city13)
	))

)
