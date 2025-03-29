Quinn Peterson | quinn.peterson@colostate.edu | Quinn-p
Joshua Weisner | joshua.weisner@colostate.edu | JoshuaWeisner
Harshith Reddy Chitreddy | reddy17@colostate.edu | HarshithChitreddy
Brendan Geraci | brendan.geraci@colostate.edu | BrendanG105

How to run our code: 
Change the input file path for each of the 3 files 
Insert start and end coordinates

A*
Cost Function:
Road Weight = .2 + Elevation difference 
OffRoad Weight = 10 + Elevation difference * Elevation Resistance
------------
Cost f(x)
Road weight = .2
Offroad weight = 5.0

Elevation resistance = 2.0
- Assuming cost of movement uphill/downhill is more when offroad than onroad  


f(x)road = road weight + elevation difference (no elevation resistance necessary 
           We are assuming that elevation resistance is negligible on road)
f(x) offroad =  off road weight + (elevation difference * elevation resistance)
------------
Test Data

From Coordinates: (3144, 6855)
To Coordinates: (1453, 8678)
Total path cost: 1322.50

From Coordinates: (3154, 6865)
To Coordinates: (1453, 8678)
Total path cost: 1222.50

From Coordinates: (2544, 7255)
To Coordinates: (1453, 8678)
Total path cost: 2356.60

------------
D* vs A* vs A* Bidirectional Search

D* is for dynamic data, which may prove more useful if live updates of the map are provided
    
    --> Like A* but only updates affected areas to avoid recalculating, which is too heavy on the               memory. Not necessary for this specific prompt.
    --> D* starts at the endpoint and recomputes the best path based on changing data.

A* was efficient at determining the fastest way from point A to B, takes time but has the best possible routing
    
    --> Cost/Value way of calculating the best way from start to end
    --> Each node(pixel) is assigned a cost; the higher the cost, the more difficult it is to traverse.         The final cost is the lowest possible cost. Meaning the most efficient way from point A to B.

A* Bidirectional Search was far faster at locating the best way from A to B when taking roads and using elevation data
    --> Cost/Value way of calculating the best way from A to B, from both A and B, meets in the middle

------------
Problem: Our team had problems with the DEM file not reading elevation data have to do with infinite numbers

Solution: Creating the mask to filter to remove infinate values
------------
Switched search algorithm from A* to bidirectional A* search, which differs from A* because it starts at both the start and endpoints and calculates the most "cost-effective" route.

Time function results based on random start, end coordinates
A* - 19.89
A* Bidirectional - 7.7 seconds
------------

