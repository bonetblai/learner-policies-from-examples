(define (problem ZTRAVEL-1-14)
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
	person13
	person14
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
	(at plane1 city9)
	(aircraft plane1)
	(fuel-level plane1 fl0)
	(at person1 city11)
	(person person1)
	(at person2 city1)
	(person person2)
	(at person3 city9)
	(person person3)
	(at person4 city0)
	(person person4)
	(at person5 city9)
	(person person5)
	(at person6 city9)
	(person person6)
	(at person7 city9)
	(person person7)
	(at person8 city12)
	(person person8)
	(at person9 city1)
	(person person9)
	(at person10 city7)
	(person person10)
	(at person11 city13)
	(person person11)
	(at person12 city0)
	(person person12)
	(at person13 city1)
	(person person13)
	(at person14 city5)
	(person person14)
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
	(at person1 city13)
	(at person2 city5)
	(at person3 city5)
	(at person4 city4)
	(at person5 city8)
	(at person6 city5)
	(at person7 city6)
	(at person8 city5)
	(at person9 city0)
	(at person10 city4)
	(at person11 city2)
	(at person12 city3)
	(at person13 city13)
	(at person14 city13)
	))

)
