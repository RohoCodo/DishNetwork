import json
import numpy as np
import random as r

def create_input(all_ing, list_ing):
    in_layer = np.zeros(len(all_ing), dtype=int)
    current = 0
    for ing in all_ing:
        if ing in list_ing:
            in_layer[current] = 1
        current += 1
    
    return in_layer

def test(data, all_ing):
    check = data[0]['ingredients']
    in_check = create_input(all_ing, check)
    print(in_check)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def main():
    categories = ['id', 'cuisine', 'ingredients', 'tp_cat', 'tp_numeric']
    with open('DishTrain.json') as f:
        data = json.load(f)

    all_ingredients = set()
    taste_profile = []
    ingredient_inputs = []
    #dish -> taste profile
    prof_cats = ["Spiciness", "Oily", "Saltiness", "Sweetness", "Temperature", "Vegan", "Vegetarian"]
    prof_max = [3, 3, 3, 3, 3, 1, 1]
    for dishes in data:
        for ing in dishes['ingredients']:
            all_ingredients.add(ing)
    
    for dishes in data:
        ingredient_inputs.append(create_input(all_ingredients, dishes['ingredients']))
        taste_profile.append(dishes['tp_numeric'])
    
    weights = [np.random.random_sample() for i in range(len(all_ingredients))]

    split_profiles = [[] for i in range(7)]
    for i in range(7):
        for j in range(len(taste_profile)):
            split_profiles[i].append(taste_profile[j][i])
    
    np_split = []
    for i in range(7):
        np_split.append(np.asarray(split_profiles[i]).reshape(len(split_profiles[i]), 1))
    ingredient_inputs = np.asarray(ingredient_inputs)
    weights = np.asarray(weights)
    bias = 0.3
    lr = 0.05

    cat_dic = {}
    for l in range(len(prof_cats)):
        cat_dic[prof_cats[l]] = prof_max[l]

    Model = MultiLayerPerceptron()
    for category in prof_cats:
        Model.add_category(category)
        Model.add_output_layer(1, category)
    
    Model.add_all_layers(10, 6, prof_cats)

    Model.fit(ingredient_inputs, taste_profile, cat_dic)





class Node():
    def __init__(self, position_in_layer, is_output_node=False):
        self.weights = []
        self.inputs = []
        self.output = None
        
        self.updated_weights = []
        self.is_output_node = is_output_node
        self.delta = None
        self.position_in_layer = position_in_layer
    
    def attach_to_output(self, nodes):
        self.output_node = nodes

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return sigmoid(x)*(1-sigmoid(x))
        
    def init_weights(self, num_input):
        for i in range(num_input + 1):
            self.weights.append(r.uniform(0,1))

    def predict(self, row):
        # Reset the inputs
        self.inputs = []
        
        # We iterate over the weights and the features in the given row
        activation = 0
        for weight, feature in zip(self.weights, row):
            self.inputs.append(feature)
            activation = activation + weight*feature
            
        self.output = self.sigmoid(activation)
        return self.output

    def update_node(self):
        self.weights = []
        for new_weight in self.updated_weights:
            self.weights.append(new_weight)            

    def calculate_update(self, learning_rate, target):
        if self.is_output_node:
            # Calculate the delta for the output
            self.delta = (self.output - target)*self.output*(1-self.output)
        else:
            # Calculate the delta
            delta_sum = 0
            # this is to know which weights this neuron is contributing in the output layer
            cur_weight_index = self.position_in_layer 
            for output_node in self.output_node:
                delta_sum = delta_sum + (output_node.delta * output_node.weights[cur_weight_index])

            # Update this neuron delta
            self.delta = delta_sum*self.output*(1-self.output)
            
            
        # Reset the update weights
        self.updated_weights = []
        
        # Iterate over each weight and update them
        for cur_weight, cur_input in zip(self.weights, self.inputs):
            gradient = self.delta*cur_input
            new_weight = cur_weight - learning_rate*gradient
            self.updated_weights.append(new_weight)
    

    
#go through each dish; put it through network; comes up with result; calc error; check if 
#ok; if not, back propagate; 


class Layer():
    def __init__(self, num_neuron, is_output_layer = False):
        self.is_output_layer = is_output_layer
        self.nodes = []
        for i in range(num_neuron):
            node = Node(i, is_output_node = is_output_layer)
            self.nodes.append(node)
    
    def attach(self, layer):
        for in_node in self.nodes:
            in_node.attach_to_output(layer.nodes)
    
    def init_layer(self, num_input):
        for node in self.nodes:
            node.init_weights(num_input)

    def predict(self, row):
        activations = [node.predict(row) for node in self.nodes]
        return activations

class MultiLayerPerceptron():
    
    def __init__(self, learning_rate = 0.01, num_iteration = 100000):
        self.layers = {}
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration

    def add_category(self, category):
        if not (category in self.layers):
            self.layers[category] = []
    
    def add_output_layer(self, num_node, category):
        self.layers[category].insert(0, Layer(num_node, is_output_layer = True))
    
    def add_all_layers(self, num_node, num_layers, categories):
        for i in categories:
            for j in range(num_layers):
                self.add_hidden_layer(num_node, i)


    def add_hidden_layer(self, num_node, category):
        # Create an hidden layer
        hidden_layer = Layer(num_node)
        # Attach the last added layer to this new layer
        hidden_layer.attach(self.layers[category][0])
        # Add this layers to the architecture
        self.layers[category].insert(0, hidden_layer)

    def update_layers(self, target):
        # Iterate over each of the layer in reverse order
        # to calculate the updated weights
        for cat in self.layers.keys():
            for layer in reversed(self.layers[cat]):
                                
                # Calculate update the hidden layer
                for node in layer.nodes:
                    node.calculate_update(self.learning_rate, target)  
        
        # Iterate over each of the layer in normal order
        # to update the weights
        for cat in self.layers.keys():
            for layer in self.layers[cat]:
                for node in layer.nodes:
                    node.update_node()
    
    def fit(self, X, y, category):
        '''
            Main training function of the neural network algorithm. This will make use of backpropagation.
            It will use stochastic gradient descent by selecting one row at random from the dataset and 
            use predict to calculate the error. The error will then be backpropagated and new weights calculated.
            Once all the new weights are calculated, the whole network weights will be updated
        '''
        num_row = len(X)
        num_feature = len(X[0]) # Here we assume that we have a rectangular matrix
        
        # Init the weights throughout each category's initial layer
        for cat in self.layers.keys():
            self.layers[cat][0].init_layer(num_feature)
        
            for i in range(1, len(self.layers[cat])):
                num_input = len(self.layers[cat][i-1].nodes)
                self.layers[cat][i].init_layer(num_input)

        # Launch the training algorithm
        for i in range(self.num_iteration):
            
            # Stochastic Gradient Descent
            r_i = r.randint(0,num_row-1)
            row = X[r_i] # take the random sample from the dataset
            yhat = self.predict(row, category)
            target = y[r_i]
            
            # Update the layers using backpropagation   
            self.update_layers(target)
            
            # At every 100 iteration we calculate the error
            # on the whole training set
            if (i+1) % 1000 == 0:
                total_error = 0
                for r_i in range(num_row):
                    row = X[r_i]
                    yhat = self.predict(row, category)
                    error = 0
                    for cat in range(len(yhat)):
                        error += (y[r_i][cat] - yhat[cat])
                    total_error = total_error + error**2
                mean_error = total_error/num_row
                print(f"Iteration {i} with error = {mean_error}")
    
    def predict(self, row, category):
        '''
            Prediction function that will take a row of input and give back the output
            of the whole neural network.
        '''
        
        # Gather all the activation in the hidden layer
        
        activations = {}
        for cat in self.layers.keys():
            activations[cat] = self.layers[cat][0].predict(row)
            for i in range(1, len(self.layers[cat])):
                activations[cat] = self.layers[cat][i].predict(activations[cat])

        outputs = []
        for cat in self.layers.keys():
            for activation in activations[cat]:
                print(activation)                           
                max_score = category[cat] + 1
                for i in range(max_score):
                    if activation < (i + .5):
                        outputs.append(i)
                        break
        print(outputs)                 
        # We currently have only One output allowed
        return outputs

if __name__ == '__main__':
    main()