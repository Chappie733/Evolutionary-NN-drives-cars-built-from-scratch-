import pygame
from enum import Enum
from math import sin, cos, pi

# a simple data structure to define the current state of the car
class FowardState(Enum):
	ACCELLERATING = 0
	BRAKING = 1
	NONE = 2

class SidewaysState(Enum):
	RIGHTWARDS = 0
	LEFTWARDS = 1
	NONE = 2

class Car:
	texture = pygame.transform.scale(pygame.image.load('res\\car.png'), (48, 24)) # the texture/image of the car
	acc = 0.2 # accelleration (pixels/tick)
	dec = 0.98 # natural decelleration (in this case 2% of the speed each frame)
	MAX_VEL = 12 # maximum velocity the car can reach
	TURN_RATE = 0.4 # the velocity at which the car turns

	def __init__(self, w, h, x=0, y=0):
		self.x, self.y = x, y
		self.w, self.h = 48,24 # width and height of the car on the screen
		self.forw_state = FowardState.NONE
		self.sidew_state = SidewaysState.NONE
		self.speed = 0
		self.rotation = 0.1

	def update(self):
		if self.forw_state == FowardState.ACCELLERATING and self.speed < self.MAX_VEL: # if the car is accellerating
			self.speed += self.acc # apply the accelleration to the speed
		elif self.forw_state == FowardState.BRAKING and self.speed > 0: # else if it's braking
			self.speed -= self.acc # apply the decelleration to the speed
		else:
			self.speed *= self.dec # otherwise apply friction

		if self.sidew_state == SidewaysState.RIGHTWARDS: # if we are turning rightwards
			self.rotation -= self.TURN_RATE*self.speed # change the rotation of the car
			if self.rotation < 0: # scale it between 0 and 360 if it goes out of that range
				self.rotation += 360
		elif self.sidew_state == SidewaysState.LEFTWARDS: # same thing but for turning leftwards
			self.rotation += self.TURN_RATE*self.speed
			if self.rotation > 360:
				self.rotation %= 360
	
		self.x += self.speed*cos(self.rotation*pi/180) # change the position based on the speed and the rotation
		self.y -= self.speed*sin(self.rotation*pi/180) # with sine and cosine

	# draw the image in the right position on the screen
	def render(self, surf, w=800, h=600, delta_x=0, delta_y=0):
		rotated = pygame.transform.rotate(self.texture, self.rotation)
		new_rect = rotated.get_rect(center=self.texture.get_rect(topleft=(int(w/2-self.w/2+delta_x), int(h/2-self.h/2+delta_y))).center)
		surf.blit(rotated, new_rect.topleft)

	# method to get the velocity of the in the x and y coordinates
	def get_direction(self):
		return (self.speed*cos(self.rotation*pi/180), -self.speed*sin(self.rotation*pi/180))