def neural_network(inputs, weight):
	prediction = weighted_sum(inputs, weight)
	return prediction

def weighted_sum(inputs, weight):
	output = 0
	for i in range(len(inputs)):
		output += inputs[i] * weights[i]
	return output

number_of_all_rounders = 3
number_of_batsmen = 4
number_of_bowlers = 3

inputs = [number_of_all_rounders, number_of_batsmen, number_of_bowlers]

weights = [0.01, 0.2, 0.03]

print(neural_network(inputs, weights))