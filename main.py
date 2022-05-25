from tkinter import Tk, Canvas, Frame, BOTH
import numpy as np
import math
import random

ZONE_SIZE = [ 2000, 2000 ]

UPDATE_MS = 10

RNG = np.random.default_rng()

MASS_SCALE = 1.5
RADIUS_SCALE = 0.5

class Element:
    def __init__(self, name, color, mass, radius, ion_energy, oxi_states):
        self.name = name
        self.color = color
        self.mass = mass
        self.radius = radius
        self.ion_energy = ion_energy
        self.oxi_states = np.asarray(oxi_states)

ELEMENTS = [
    Element("Hydrogen", "#11f", 1, 120, 14, [1, -1]),
    Element("Carbon", "#555", 12, 170, 11, [4, 2, -4]),
    Element("Nitrogen", "#f11", 14, 155, 14, [5, 4, 3, 2, 1, -1, -2, -3]),
    Element("Oxygen", "#1d1", 15, 152, 13, [-2]),
    Element("Phosphorus", "#511", 30, 180, 10, [5, 3, -3]),
    Element("Sulfur", "#aa1", 32, 180, 10, [6, 4, -2])
]

def randPos():
    return np.asarray([
        RNG.uniform(0, ZONE_SIZE[0]),
        RNG.uniform(0, ZONE_SIZE[1])])

def randVel(mag):
    return np.asarray([
        RNG.random() * mag - mag / 2,
        RNG.random() * mag - mag / 2])

class Atom:
    def __init__(self, canvas, pos=None, element=None):
        self.pos = randPos() if pos == None else pos
        self.vel = randVel(100)
        
        self.element = RNG.choice(ELEMENTS) if element == None else element

        self.oxi = 0

        self.bonds = []

        self.body = canvas.create_oval(
            0, 0, 0, 0, fill=self.element.color)

        self.orbit = canvas.create_oval(
            0, 0, 0, 0, outline=self.element.color)

        self.label = canvas.create_text(0, 0,
            text=self.element.name[0], font=("Arial 12"), fill="white")

        self.oxtxt = canvas.create_text(0, 0,
            text=self.oxi, font=("Arial 8"), fill="white")

    def draw(self, canvas, pan):
        self.m0 = self.pos - self.element.mass * MASS_SCALE
        self.m1 = self.pos + self.element.mass * MASS_SCALE

        canvas.coords(self.body,
            self.m0[0] + pan[0], self.m0[1] + pan[1],
            self.m1[0] + pan[0], self.m1[1] + pan[1])

        self.r0 = self.pos - self.element.radius * RADIUS_SCALE
        self.r1 = self.pos + self.element.radius * RADIUS_SCALE

        canvas.coords(self.orbit,
            self.r0[0] + pan[0], self.r0[1] + pan[1],
            self.r1[0] + pan[0], self.r1[1] + pan[1])

        canvas.coords(self.label, self.pos[0] + pan[0], self.pos[1] + pan[1])
        canvas.coords(self.oxtxt, self.pos[0]+10 + pan[0], self.pos[1]+10 + pan[1])

        canvas.itemconfigure(self.oxtxt, text=self.oxi)

    def update(self, paused):
        if not paused:
            next_pos = self.pos + self.vel

            def bounce(a):
                if next_pos[a] < 0 or next_pos[a] > ZONE_SIZE[a]:
                    self.vel[a] = -self.vel[a]

            bounce(0)
            bounce(1)

            for bond in self.bonds:
                other = bond.other(self)

                if not self.orbiting(other):
                    delta = other.pos - self.pos
                    self.pos += delta / np.power(np.linalg.norm(delta), 0.5)

            self.pos += self.vel
            self.vel *= 0.98

        def clip(a):
            if self.pos[a] < 0: self.pos[a] = 0
            if self.pos[a] > ZONE_SIZE[a]: self.pos[a] = ZONE_SIZE[a]

        clip(0)
        clip(1)
        
    def orbiting(self, other):
        rad = (self.element.radius * RADIUS_SCALE + 
              other.element.radius * RADIUS_SCALE)
        return np.linalg.norm(self.pos - other.pos) < rad

    def disconnected(self, other):
        rad = (self.element.radius * RADIUS_SCALE + 
              other.element.radius * RADIUS_SCALE)
        return np.linalg.norm(self.pos - other.pos) > rad*2 + rad*self.stability()

    def colliding(self, other):
        rad = (self.element.mass * MASS_SCALE + 
              other.element.mass * MASS_SCALE)
        return np.linalg.norm((self.pos + self.vel) -
            (other.pos + self.vel)) < rad

    def inside(self, pos):
        rad = self.element.radius * RADIUS_SCALE
        return np.linalg.norm(self.pos - pos) < rad

    def bonded(self, other):
        for bond in self.bonds:
            if bond.a == other or bond.b == other:
                return True

        return False

    def reflect(self, other):
        self.pos += (self.pos - other.pos) * 0.05

    def stability(self, ex=0):
        a_score = np.min(np.abs(self.element.oxi_states - self.oxi + ex))
        return 1 / (a_score + 1)


class Bond:
    def __init__(self, a, b):
        self.a = a
        self.b = b

        self.ex_a = 0
        self.ex_b = 0

        self.line = None

        maxes = [np.max(a.element.oxi_states), np.max(b.element.oxi_states)]
        mins = [np.min(a.element.oxi_states), np.min(b.element.oxi_states)]

        oxrange = [ np.max(mins), np.min(maxes) ]

        def trim(arr, oxi):
            if oxi > 0: return arr[(arr > oxi)]
            if oxi < 0: return arr[(arr < oxi)]

            return arr

        a_states = trim(a.element.oxi_states, a.oxi)
        b_states = trim(b.element.oxi_states, b.oxi)

        if a_states.size != 0 and b_states.size != 0:
            self.ex_a = a_states[np.argmax(np.abs(a_states))] - a.oxi
            self.ex_b = b_states[np.argmax(np.abs(b_states))] - b.oxi

            if np.abs(self.ex_a) < np.abs(self.ex_b):
                self.ex_b = -self.ex_a
            elif np.abs(self.ex_a) > np.abs(self.ex_b):
                self.ex_a = -self.ex_b

        # calculate how stable the bond is
        if self.ex_a == self.ex_b: 
            self.stability = 0
            return

        self.stability = (a.stability(self.ex_a) + b.stability(self.ex_b)) / 2

    def other(self, atom):
        return self.b if self.a == atom else self.a

    def draw(self, canvas, pan):
        if not self.line: return

        canvas.itemconfig(self.line, fill=
            "#%02x%02x%02x" % 
            (255, int(self.stability * 255), int(self.stability * 255)))

        canvas.coords(self.line, 
            self.a.pos[0] + pan[0], self.a.pos[1] + pan[1],
            self.b.pos[0] + pan[0], self.b.pos[1] + pan[1])

    def update(self):
        self.stability = (self.a.stability() + self.b.stability()) / 2

    def apply(self, canvas, bonds):
        self.a.oxi += self.ex_a
        self.b.oxi += self.ex_b

        self.a.bonds.append(self)
        self.b.bonds.append(self)

        self.line = canvas.create_line(
            0, 0, 0, 0, fill="black")

        bonds.append(self)

    def remove(self, canvas, bonds):

        self.a.oxi -= self.ex_a
        self.b.oxi -= self.ex_b

        self.stability = 0

        self.a.bonds.remove(self)
        self.b.bonds.remove(self)
        
        canvas.delete(self.line)

        bonds.remove(self)

class Simulator(Frame):
    def __init__(self):
        super().__init__()

        self.master.title("Particle Simulator")
        self.pack(fill=BOTH, expand=1)

        self.canvas = Canvas(self)
        self.canvas.configure(bg='black')
        self.canvas.focus_set()

        self.bounds = self.canvas.create_rectangle(
            0, 0, ZONE_SIZE[0], ZONE_SIZE[1],
            outline="white")

        self.zoom = 1.0
        self.pan = np.asarray([0, 0])

        self.atoms = []
        self.bonds = []

        self.selAtom = None

        for x in range(64):
            self.atoms.append(Atom(self.canvas))

        self.paused = False

        self.update()

    def zoomIn(self, event):
        if self.zoom < 2.0: self.zoom += 0.1
        else: self.zoom = 2.0

    def zoomOut(self, event):
        if self.zoom > 0.1: self.zoom -= 0.1
        else: self.zoom = 0.1

    def moveLeft(self, event): self.pan[0] += 1 / self.zoom * 32
    def moveRight(self, event): self.pan[0] -= 1 / self.zoom * 32
    def moveUp(self, event): self.pan[1] += 1 / self.zoom * 32
    def moveDown(self, event): self.pan[1] -= 1 / self.zoom * 32

    def toWorld(self, coords):
        pos = np.asarray([float(coords.x), float(coords.y)])
        return (pos - self.pan) / self.zoom

    def selectAtom(self, event):
        pos = self.toWorld(event)
        for atom in self.atoms:
            if atom.inside(pos):
                self.selAtom = atom
    
    def moveAtom(self, event):
        if self.selAtom == None: return

        pos = self.toWorld(event)
        dif = pos - self.selAtom.pos

        if self.paused:
            self.selAtom.pos += dif * 0.5
        else:
            self.selAtom.vel = dif * 0.5

    def deselectAtom(self, event):
        self.selAtom = None

    def resetAtom(self, event):
        pos = self.toWorld(event)

        for atom in self.atoms:
            if atom.inside(pos):
                for i, bond in enumerate(self.bonds):
                    if bond in atom.bonds:
                        bond.remove(self.canvas, self.bonds)

                return

    def pause(self, event):
        self.paused = not self.paused

    def update(self):

        self.canvas.coords(self.bounds,
            self.pan[0], self.pan[1],
            self.pan[0] + ZONE_SIZE[0], self.pan[1] + ZONE_SIZE[1])

        for atom in self.atoms:
            for other_atom in self.atoms:
                if atom != other_atom:
                    if atom.colliding(other_atom):
                        atom.reflect(other_atom)
                        other_atom.reflect(atom)

                    if (not atom.bonded(other_atom) and
                        atom.orbiting(other_atom)):

                        b = Bond(atom, other_atom)

                        if b.stability > 0:
                            b.apply(self.canvas, self.bonds)

                    if (atom.bonded(other_atom) and
                        atom.disconnected(other_atom)):
                        for bond in atom.bonds:
                            if bond.other(atom) == other_atom:
                                bond.remove(self.canvas, self.bonds)

            atom.update(self.paused)
            atom.draw(self.canvas, self.pan)

        for bond in self.bonds:
            bond.update()
            bond.draw(self.canvas, self.pan)

        if self.selAtom != None:
            self.selAtom.vel *= 0.75

        self.canvas.pack(fill=BOTH, expand=1)

        self.canvas.scale("all", self.pan[0], self.pan[1], self.zoom, self.zoom)

        self.after(UPDATE_MS, self.update)


def main():
    root = Tk()
    sim = Simulator()
    root.geometry("1600x900")

    #keyboard controls
    root.bind("<space>", sim.pause)
    root.bind("q", sim.zoomIn)
    root.bind("e", sim.zoomOut)
    root.bind("w", sim.moveUp)
    root.bind("s", sim.moveDown)
    root.bind("a", sim.moveLeft)
    root.bind("d", sim.moveRight)

    #mouse controls
    root.bind("<ButtonPress-1>", sim.selectAtom)
    root.bind("<B1-Motion>", sim.moveAtom)
    root.bind("<ButtonRelease-1>", sim.deselectAtom)
    root.bind("<Button-3>", sim.resetAtom)
    
    root.mainloop()

if __name__ == '__main__':
    main()
