import pygame
import os
import numpy as np
from car import *
from genome import *
from layer import Layer, tanh
from math import sqrt, cos, sin, pi
import pickle

w,h = 800, 600 # size of window (in pixels)
map_w, map_h = 4000, 2250 # size of map (in pixels)

# LOADING ALL OF THE IMAGES
# used to determine where it is drivable and where it isn't
map_texture = pygame.transform.scale(pygame.image.load('res\\map_1.png'), (map_w, map_h)) 
# actual image of the map, the one drawn on the screen
map_render = pygame.transform.scale(pygame.image.load('res\\map_render.png'), (map_w, map_h))
# map with every invisible line that are used to determine how far a car has gotten
rewards_texture = pygame.transform.scale(pygame.image.load('res\\rewards.png'), (map_w, map_h))

# euclidean distance between two points... duh
def get_euc(a, b):
	return sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def norm(x):
	return 1 if x >= 0 else -1

# this function returns the distance to the closest obstacle ahead of the car (only ahead)
# at first I wanted to do a binary search to approximate the position where the bound starts
# but this also works and it's efficient enough either way and it avoids complicacies
def get_distance(car, acceptable_error, step_size=2):
	global w, h
	r = car.rotation*pi/180
	curr_dist = 0

	while get_euc((0, 0), (curr_dist*cos(r), -curr_dist*sin(r))) < 300:
		if map_texture.get_at((int(w/2+car.x+curr_dist*cos(r)), int(h/2+car.y-curr_dist*sin(r))))[0] == 0:
			break
		curr_dist += step_size

	return get_euc((0,0), (curr_dist*cos(r), -curr_dist*sin(r)))

# [front, left, right], yes this is absolutely fucking and utterly retarded but it works and I couldn't get it to work any other way
# this function returns the distances to the bounds of the road
def get_distances(car, map_pos, acceptable_error=5):
	global map_texture
	results = [get_distance(car, acceptable_error)]
	car.rotation += 90
	results.append(get_distance(car, acceptable_error))
	car.rotation -= 180
	results.append(get_distance(car, acceptable_error))
	car.rotation += 90
	return results

# checks if the car is crashing in the current frame, more practically if the car is on a black pixel
# in the image that represents in white the "driveable area"
def is_crashing(car, map_pos):
	global map_texture, w, h
	try:
		if map_texture.get_at((int(w/2-map_pos[0]), int(h/2-map_pos[1])))[0] == 0 or map_texture.get_at((int(w/2-map_pos[0]+car.w/2), int(h/2-map_pos[1])))[0] == 0:
			return True
		elif map_texture.get_at((int(w/2-map_pos[0]), int(h/2-map_pos[1]+car.h/2)))[0] == 0 or map_texture.get_at((int(w/2-map_pos[0]+car.w/2), int(h/2-map_pos[1]+car.h/2)))[0] == 0:
			return True
	except:
		pass
	return False

# determines whether a checkpoint has been passed, this is checked by seeing if the car passed trought a black pixel
# in the image with every checkpoint signed in black, the car also has to be moving or else it could just learn to stand still
# on a checkpoint/reward point
def passed_checkpoint(car, map_pos):
	try:
		return rewards_texture.get_at((int(w/2-map_pos[0]+car.w/2), int(h/2-map_pos[1]+car.h/2)))[0] == 0 and car.speed != 0
	except:
		return False

def main(gens=150, fitness_threshold=380):
	POPULATION = 40 # cars per generation
	MTPG = 3 # time after which the generation ends if no new fitness is gained by any individual
	delta = 0.2
	# if delta is d, a genome is part of the top species that dictates the next generation if its fitness is between [max*(1-d), max],
	# where max is the maximum fitness in the current generation

	win = pygame.display.set_mode((w,h))
	surf = pygame.Surface((w,h))
	running = True
	clock = pygame.time.Clock()
	fps = 60

	selected_car_index = 0 # index of the car the program's focusing on
	cars = [Car(w,h, x=380, y=325) for _ in range(POPULATION)] # generate the cars
	genomes = [Genome(net_shape=[Layer(3), Layer(4, activation=tanh), Layer(2)]) for _ in range(POPULATION)] # generate the AIs
	max_fitness = 0 # the maximum efficiency a car has reached
	alive = POPULATION
	generations = 1 # the current generation

	generation_time = 0 # time since the last fitness has been gained

	best = [] # best genomes/AIs of each generation

	while running:
		clock.tick()
		generation_time += clock.get_time()
		surf.fill((10,132,12))

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_w:
					selected_car_index += 1 if selected_car_index != len(cars)-1 else -selected_car_index

		surf.blit(map_render, (-cars[selected_car_index].x, -cars[selected_car_index].y))
		for i, car in enumerate(cars): # for every i(th) car
			if not genomes[i].alive: # if the car's dead/it has crashed, don't do anything
				continue
			distances = np.array(get_distances(car, (-car.x, -car.y))) 
			
			predictions = genomes[i].nn.predict(distances) # let the ai decide what to do
			if passed_checkpoint(car, (-car.x, -car.y)): # if a car passed a checkpoint/reward line
				if genomes[i].fsr >= 8: # and it hasn't passed one in the last 8 frames
					genomes[i].fitness += 2 # reward it
					generation_time = 0 # reset the timer to wipe the generation
					genomes[i].fsr = 0
				if genomes[i].fitness > max_fitness:
					max_fitness = genomes[i].fitness # if a new AI surpasses the best one, update the best ranking
					selected_car_index = i # always look at the car that's doing the best

			# apply the numerical predictions of the AI to the car
			if predictions[0] <= -0.5:
				car.sidew_state = SidewaysState.LEFTWARDS
			elif predictions[0] >= 0.5:
				car.sidew_state = SidewaysState.RIGHTWARDS
			else:
				car.sidew_state = SidewaysState.NONE

			if predictions[1] >= 0.5:
				car.forw_state = FowardState.ACCELLERATING
			elif predictions[1] <= -0.8: # the threshold for the AI to brake is much higher because I didn't want it to brake much
				car.forw_state = FowardState.BRAKING
			else:
				car.forw_state = FowardState.NONE

			car.update()
			if is_crashing(car, (-car.x, -car.y)): # if the car crashed/ran out of the track
				genomes[i].fitness -= 5 # "punish" it, by making it lose "ranking" (fitness)
				genomes[i].alive = False # set it to dead
				alive -= 1 # decrease the amount of cars that are still alive/driving

			# draw the car on the screen
			car.render(surf, delta_x=car.x-cars[selected_car_index].x, delta_y=car.y-cars[selected_car_index].y)
			genomes[i].fsr += 1 # update the frames since the last reward of the current car

		
		# if every car has crashed, or enough time has passed since any car passed a checkpoint (made progress)
		if alive == 0 or generation_time >= MTPG*1000:
			best_genomes = []
			print('-'*15, "Generation " + str(generations), '-'*15)
			print("max fitness: " + str(max_fitness))

			# select the best genomes/AIs
			curr_best = genomes[0]
			for genome in genomes:
				if genome.fitness >= (max_fitness-5)*(1-delta):
					best_genomes.append(genome)
				if genome.fitness >= curr_best.fitness:
					curr_best = genome
			best.append(curr_best)

			# build the next generation with mutations from the best AIs of the current one
			genomes.clear()
			for genome in best_genomes:
				genomes += genome.get_mutations(mutations=int(POPULATION/len(best_genomes)))
			genomes += curr_best.get_mutations(mutations=POPULATION-len(genomes))

			# reset the positions, speed and rotations of all the cars
			for i in range(len(cars)):
				cars[i].x, cars[i].y = 380, 325
				cars[i].rotation = 0.1
				cars[i].speed = 0

			selected_car_index = 0
			alive = POPULATION # reset the amount of active cars to the amount of cars generated
			generations += 1 # increase the generation counter by 1
			# if enough generations have passed or an AI has reached a sufficiently high fitness/effectiveness
			if generations > gens or max_fitness >= fitness_threshold: 
				with open('best_genomes', "wb") as f: # save the best AI's from each generation in a file
					pickle.dump(best, f)
				running = False # and stop the program
			generation_time = 0 # reset the time since the last individual has gained fitness/been rewarded
			max_fitness = 0 # and reset the maximum fitness of the current generation to 0

		win.blit(surf, (0,0)) # draw everything on the screen
		pygame.display.flip() # and flip the buffer to make it visible

if __name__ == '__main__':
	pygame.init()
	main(gens=150)
	pygame.quit()
