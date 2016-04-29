import numpy as np
from skyfield.api import Loader
from werkzeug.contrib.cache import RedisCache
from astropy import units as u
import os

cache = RedisCache(host='localhost', port=6379)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#cache = SimpleCache()

G = 6.67408e-11 # Gravitational constant (m^3/(kg s^2)
Me = 5.972e24 # mass of earth (kg)
Mm = 7.348e22 # mass of moon (kg)
Re = 6.371e6 # Radius of earth (m)
Rm = 1.737e6 # Radius of moon (m)
vm = 1.022e3 # velocity of the moon (m/s)
dt = 1e3 # time step to be used in simulation (s)

OneMonth = 3.e6 # one month in seconds
EarthToMoon = 370e6

load = Loader(os.path.join(BASE_DIR, 'data'))
Planets = load('de421.bsp')
ts = load.timescale()
now = ts.now()

sun = Planets['sun'].at(now)
Origin = sun.position.to(u.m).value
Ref_frame = sun.velocity.to(u.m/u.s).value

Center_Body_Name = 'earth'

#{name:{mass: 1e24 kg, diameter: km}, ...}
# from http://nssdc.gsfc.nasa.gov/planetary/factsheet/
planet_properties = {
    "sun":{"mass":1.99e6,"diameter":1e6},
    "mercury":{"mass":0.33,"diameter":4879},
    "venus":{"mass":4.87,"diameter":12104},
    "earth":{"mass":5.97,"diameter":12756},
    "moon":{"mass":0.073,"diameter":3475},
    "mars":{"mass":0.642,"diameter":6792},
    "jupiter":{"mass":1898,"diameter":142984},
    "saturn":{"mass":568,"diameter":120536},
    "uranus":{"mass":86.8,"diameter":51118},
    "neptune":{"mass":102,"diameter":49528},
    "pluto":{"mass":0.0146,"diameter":2370},
    }

class CelestialBody(object):    


    def __init__(self, coords, velocity, m, R, name="object"):
        """initial conditions of body

        Args
        ----
        coords (float, float): x, y coordinates of body (m)
        velocity (float, float): vx, vy coordinates of body (m/s)
        m (float): mass of body (kg)
        R (float): radius of body
        """
        self.fuel_used = 0
        self.name = name
        self.x, self.y = float(coords[0]), float(coords[1])
        self.vx, self.vy = float(velocity[0]), float(velocity[1])
        self.m = float(m)
        self.R = R

    @property
    def coords(self):
        """Current position of body

        Returns 
        -------
        (x, y) (float, float): position in m
        """
        return self.x, self.y

    @property
    def velocity(self):
        """Current velocity of body

        Returns
        -------
        (vx, vy) (float, float): velocity in  m/s
        """
        return self.vx, self.vy

    @property
    def speed(self):
        """Speed of moon i.e. magnitude of velocity

        Returns
        -------
        speed (float): m/s
        """
        vx, vy = self.velocity
        return np.sqrt(vx**2 + vy**2)

    def distance_from(self, body):
        """Find the distance between this body and a different one
        (m)
        """
        dx = self.x - body.x
        dy = self.y - body.y
        return np.sqrt( dx**2 + dy**2  )

    @property
    def kinetic_energy(self):
        """Kinetic energy of this body 
        J = kg*(m/s)**2
        """
        return .5*self.m*self.speed**2


    def potential_energy(self, body):
        """What is the potential energy of this body, given the gravitational 
        field exerted on it by another body 

        J = kg*(m/s)**2
        """

        r = self.distance_from(body)
        return -G*body.m*self.m/r

    def gravitational_field(self, x, y):
        """Find the acceleration imposed my this celestial body at a given point
        in space

        (m/s**2)

        Args
        ----
        coord (float, float): x,y position in space

        Returns
        -------
        acceleration (float, float): ax xhat , ay yhat
        """
        dx = x - self.x
        dy = y - self.y

        r = np.sqrt( dx**2 + dy**2  )
        a = - G * self.m / r**3 
        return a*dx, a*dy 

    def move_through(self, system, dt=dt):
        """RK 4th order movement 

        Move this body 1 timestep through a system of other celestial bodies due 
        to the force exerted by them

        x(dt) = x(0) + 1/6 (v1 + 2*(v2 + v3) + v4)*dt
        v(dt) = v(0) + 1/6 (a1 + 2*(a2 + a3) + ay)*dt

        Velocity 
        --------
        v1 = xdot(x,t) 
        v2 = xdot(x,t) + a1*dt/2 
        v3 = xdot(x,t) + a2*dt/2 
        v4 = xdot(x,t) + a3*dt

        Acceleration
        ------------
        a1 = xddot(x)
        a2 = xddot(x+v1*dt/2)
        a3 = xddot(x+v2*dt/2)
        a4 = xddot(x+v3*dt)

        Args
        ----
        system [CelestialBody]
        """
        
        bodies = self.other_bodies(system)

        x, y = self.coords
        a1x, a1y = effective_gravitational_field(x, y, bodies)
        v1x, v1y = self.velocity

        a2x, a2y = effective_gravitational_field(x+v1x*dt/2., y+v1y*dt/2., bodies)
        v2x = v1x + a1x*dt/2.
        v2y = v1y + a1y*dt/2.

        a3x, a3y = effective_gravitational_field(x+v2x*dt/2., y+v2y*dt/2., bodies)
        v3x = v1x + a2x*dt/2.
        v3y = v1y + a2y*dt/2.

        a4x, a4y = effective_gravitational_field(x+v3x*dt, y+v3y*dt, bodies)
        v4x = v1x + a3x*dt
        v4y = v1y + a3y*dt

        self.vx = v1x + (a1x + 2*(a2x+a3x) + a4x)*dt/6.
        self.vy = v1y + (a1y + 2*(a2y+a3y) + a4y)*dt/6.

        self.x = x + (v1x + 2*(v2x+v3x) + v4x)*dt/6.
        self.y = y + (v1y + 2*(v2y+v3y) + v4y)*dt/6.

    def other_bodies(self, system):
        """
        """
        return [body for body in system if body != self]

    def thruster(self, thruster_direction, detonation_number=10000):
        """Nuclear propulsion D2 thruster 

        Args
        -----
        thruster_direction (float): direction to fire the thruster (rad)
        detonation_number (int): how many microbombs to detonate
        """
        E = .1 * 5.e12 # 10% usable Energy form dd micro-bomb (J)

        v = detonation_number * np.sqrt(2.*E/self.m)

        self.vx += v*np.cos(thruster_direction)
        self.vy += v*np.sin(thruster_direction)


    def check_collisions(self, system):
        """See if this object collides with anything in the system
        
        Args
        ----
        system [CelestialBody]

        Returns
        ------
        bool True if collision
        """

        bodies = self.other_bodies(system)

        for body in bodies:
            if self.distance_from(body) < (self.R + body.R):
                print "The {} will collide with the {}".format(self.name, body.name)
                return True
            else:
                return False

    def data_dump(self):
        """Dump out all the necessary data so that it can be loaded into a clone

        Returns
        -------

        """
        return {"coords":self.coords,
                "velocity":self.velocity,
                "m":self.m,
                "R":self.R,
                "name":self.name
                }

    def in_bounds(self):
        """Check to see if the body is within bounds of the system
        """
        return np.sqrt( self.x**2 + self.y**2 ) < 1e15


def effective_gravitational_field(x, y, bodies):
    """Calculate the effective gravitational field from a system of 
    celestial bodies

    Args
    ----
    x (float): x position to calculate the gravitational acceleration 
    y (float): y position to calculate the gravitational acceleration 
    bodies [CelestialBody]: array of celestial bodies to use in the calculation

    Return
    ------
    (ax, ay) (float, float): acceleration at point (x,y) due to the system
    """

    ax = 0
    ay = 0

    for body in bodies:
        axtmp, aytmp = body.gravitational_field(x,y)
        ax+=axtmp
        ay+=aytmp

    return ax, ay


def predict_trajectory(data, tau=2*OneMonth):
    """Perdict the behavior of the system given the initial conditions from a 
    Celestial body data dump

    Args
    ----
    data ([dict]): celestial body data dump. Can be used to clone body
    tau (flow): amount of time to look into the future (seconds)

    Returns
    -------
    trajectories (dict): {"body name": }
    """
    running = cache.get("running")
    if not running:
        return

    system = [CelestialBody(**dump) for dump in data] 

    nt = int(tau/dt)
    
    trajectories = {}

    for i in range(nt):
        for body in system:
            if body.name != Center_Body_Name:
                body.move_through(system, dt=3*dt)

                if body.check_collisions(system):
                    print "*"*50
                    #print "Collision in T-{} days".format(i*dt/86400.)
                    print "*"*50
                    return trajectories

                if body.name == "asteroid":
                    try:
                        trajectories[body.name].append(body.coords)
                    except KeyError:
                         trajectories[body.name] = [body.coords]
        
    return trajectories

def set_cache(key, val):
    """Set cache
    """
    cache.set(key, val, 3600)

def move_system(system):
    """
    """
    for body in system:
        if body.name != 'sun':
            body.move_through(system, dt=dt)


def generate_celestial_body(name):
    """
    """

    skyfield_planet = Planets[name]

    x, y, z = skyfield_planet.at(now).position.to(u.m).value - Origin
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y/x)
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    vx, vy, vz = skyfield_planet.at(now).velocity.to(u.m/u.s).value - Ref_frame
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    vx = v*np.sin(theta)
    vy = v*np.cos(theta)

    p = planet_properties[name]
    m = p['mass']*1e24
    R = p['diameter']*.5e3
    return CelestialBody( (x, y), (vx,vy), m, R, name)


def simulation():

    """
    sun = CelestialBody((0, 0), (0,0), 
                    planet_properties["sun"]['mass']*1e24, 
                    planet_properties["sun"]['diameter']*.5e3, 
                    'sun')
    mercury = generate_celestial_body("mercury") 
    venus = generate_celestial_body("venus") 
    earth = generate_celestial_body("earth") 
    moon = generate_celestial_body("moon") 

    theta = np.arctan(1./2.)
    v = 100.e2
    vx = v*np.cos(theta)
    vy = v*np.sin(theta)

    asteroid = CelestialBody((-4.e11, 0), (0,v), 10.e18, 10*Re, "asteroid")

    system = [sun, mercury, venus, earth, moon, asteroid]
    """
    
    theta = np.arctan(1./3.)
    v = 8.e2
    vx = v*np.cos(theta)
    vy = v*np.sin(theta)

    earth = CelestialBody( (0,0), (0,0), Me, Re, 'earth')
    moon = CelestialBody( (EarthToMoon,0), (0,vm), Mm, Rm, 'moon')
    asteroid = CelestialBody( (-2*EarthToMoon,0), (vx,vy), 10.e18, Rm, 'asteroid')

    system = [earth, moon, asteroid]

    set_cache("data", list(map(lambda body: body.data_dump(), system)))
    
    run = 'true'
    set_cache("running", run)

    while run == 'true':

        for n in range(10):
            #mercury.move_through([sun]) 
            #venus.move_through([sun]) 
            #earth.move_through([sun]) 
            moon.move_through(system) 
            asteroid.move_through(system)


            if asteroid.check_collisions(system):
                print "\n\nGame over. You crashed the rock."
                break
            elif not asteroid.in_bounds():
                print "\n\nGame over. You lost the Rock."
                break


            thruster_direction = cache.get("thruster_direction")
            if thruster_direction:
                asteroid.thruster(float(thruster_direction))
                cache.set("thruster_direction", None, 3600)

        set_cache("data", list(map(lambda body: body.data_dump(), system)))
        run = cache.get("running")

    set_cache("running", 'false')

