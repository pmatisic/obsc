import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Spremnik:
    def __init__(self, sirina):
        self.sirina = sirina

class Baterija:
    def __init__(self, duzina, sirina, x, y, rotacija):
        self.duzina = duzina
        self.sirina = sirina
        self.x = x
        self.y = y
        self.rotacija = rotacija

class Jedinica:
    def __init__(self, baterije):
        self.baterije = baterije

    def izracunajFitness(self, spremnik):
        ukupnaPovrsina = 0
        for i, baterija in enumerate(self.baterije):
            ukupnaPovrsina += baterija.duzina * baterija.sirina
            if (baterija.x + baterija.duzina > spremnik.sirina) or (baterija.y + baterija.sirina > spremnik.sirina):
                return 0
            for j, drugaBaterija in enumerate(self.baterije):
                if i != j:
                    if (baterija.x < drugaBaterija.x + drugaBaterija.duzina and
                            baterija.x + baterija.duzina > drugaBaterija.x and
                            baterija.y < drugaBaterija.y + drugaBaterija.sirina and
                            baterija.y + baterija.sirina > drugaBaterija.y):
                        return 0
        return ukupnaPovrsina

    def provjeriPreklapanje(self, indeks):
        b1 = self.baterije[indeks]
        for i, b2 in enumerate(self.baterije):
            if i != indeks:
                if (b1.x < b2.x + b2.duzina and
                        b1.x + b1.duzina > b2.x and
                        b1.y < b2.y + b2.sirina and
                        b1.y + b1.sirina > b2.y):
                    return True
        return False

    @staticmethod
    def rotirajTocku(tocka, kut):
        x, y = tocka
        cosKut = np.cos(kut)
        sinKut = np.sin(kut)
        return cosKut * x - sinKut * y, sinKut * x + cosKut * y

    @staticmethod
    def normalniVektoriElipse(a, b, rotacija):
        u = (a * np.cos(rotacija), b * np.sin(rotacija))
        v = (-b * np.sin(rotacija), a * np.cos(rotacija))
        return u, v

    @staticmethod
    def projekcijaTocakaNaOs(tocaka, os):
        return [np.dot(tocka, os) for tocka in tocaka]

    @staticmethod
    def intervalPreklapanja(proj1, proj2):
        return max(proj1) >= min(proj2) and max(proj2) >= min(proj1)

    def mutiraj(self, spremnik):
        indeks = random.randint(0, len(self.baterije) - 1)
        baterija = self.baterije[indeks]
        brojPokusaja = 0
        tockeNaElipsi = [(np.cos(t) * baterija.duzina, np.sin(t) * baterija.sirina) for t in np.linspace(0, 2 * np.pi, 100)]

        while True:
            baterija.x = random.randint(0, 1000 * (spremnik.sirina - baterija.duzina)) / 1000.0
            baterija.y = random.randint(0, 1000 * (spremnik.sirina - baterija.sirina)) / 1000.0
            baterija.rotacija = random.randint(0, 360000) / 1000.0
            rotiraneTocke = [Jedinica.rotirajTocku(tocka, np.radians(baterija.rotacija)) for tocka in tockeNaElipsi]
            preklapanje = False

            for drugaBaterija in self.baterije:
                if drugaBaterija != baterija:
                    tockeNaDrugojElipsi = [(np.cos(t) * drugaBaterija.duzina, np.sin(t) * drugaBaterija.sirina) for t in np.linspace(0, 2 * np.pi, 100)]
                    rotiraneTockeDrugeBaterije = [Jedinica.rotirajTocku(tocka, np.radians(drugaBaterija.rotacija)) for tocka in tockeNaDrugojElipsi]
                    osi = Jedinica.normalniVektoriElipse(baterija.duzina, baterija.sirina, np.radians(baterija.rotacija)) + \
                          Jedinica.normalniVektoriElipse(drugaBaterija.duzina, drugaBaterija.sirina, np.radians(drugaBaterija.rotacija))
                    for os in osi:
                        projekcijaPrve = Jedinica.projekcijaTocakaNaOs(rotiraneTocke, os)
                        projekcijaDruge = Jedinica.projekcijaTocakaNaOs(rotiraneTockeDrugeBaterije, os)
                        if not Jedinica.intervalPreklapanja(projekcijaPrve, projekcijaDruge):
                            preklapanje = False
                            break
                        preklapanje = True

            if not preklapanje or brojPokusaja > 100:
                break
            
            brojPokusaja += 1

    @staticmethod
    def krizaj(roditelj1, roditelj2):
        polovina = len(roditelj1.baterije) // 2
        dijeteBaterije = roditelj1.baterije[:polovina] + roditelj2.baterije[polovina:]
        return Jedinica(dijeteBaterije)

    def __str__(self):
        baterijeInfo = "\n".join([f"Baterija na poziciji x={b.x}, y={b.y}, rotacija={b.rotacija}" for b in self.baterije])
        return f"Jedinica s ukupnom površinom {self.izracunajFitness(spremnik)}\n{baterijeInfo}"

class GenetskiAlgoritam:
    def __init__(self, velicinaPopulacije, brojGeneracija, stopaMutacije, stopaKrizanja, spremnik, baterije):
        self.velicinaPopulacije = velicinaPopulacije
        self.brojGeneracija = brojGeneracija
        self.stopaMutacije = stopaMutacije
        self.stopaKrizanja = stopaKrizanja
        self.spremnik = spremnik
        self.baterije = baterije
        self.populacija = self.inicijalizirajPopulaciju()

    def inicijalizirajPopulaciju(self):
        populacija = []
        for _ in range(self.velicinaPopulacije):
            jedinicaBaterije = []
            for b in self.baterije:
                maksX = self.spremnik.sirina - b.duzina
                maksY = self.spremnik.sirina - b.sirina
                x = random.uniform(0, maksX / self.spremnik.sirina)
                y = random.uniform(0, maksY / self.spremnik.sirina)
                jedinicaBaterije.append(Baterija(b.duzina, b.sirina, x, y, b.rotacija))
            populacija.append(Jedinica(jedinicaBaterije))
        return populacija

    def pokreni(self):
        for generacija in range(self.brojGeneracija):
            dobrote = [jedinica.izracunajFitness(self.spremnik) for jedinica in self.populacija]

            if sum(dobrote) == 0:
                roditelj1 = random.choice(self.populacija)
                roditelj2 = random.choice(self.populacija)
            else:
                roditelj1 = random.choices(self.populacija, weights=dobrote)[0]
                roditelj2 = random.choices(self.populacija, weights=dobrote)[0]

            if random.random() < self.stopaKrizanja:
                dijete = Jedinica.krizaj(roditelj1, roditelj2)
                self.populacija.append(dijete)

            for jedinica in self.populacija:
                if random.random() < self.stopaMutacije:
                    jedinica.mutiraj(spremnik)

            dobrote = [jedinica.izracunajFitness(self.spremnik) for jedinica in self.populacija]

            if sum(dobrote) == 0:
                self.populacija = random.choices(self.populacija, k=self.velicinaPopulacije)
            else:
                self.populacija = random.choices(self.populacija, weights=dobrote, k=self.velicinaPopulacije)

        dobrote = [jedinica.izracunajFitness(self.spremnik) for jedinica in self.populacija]
        najboljeRjesenje = max(self.populacija, key=lambda x: x.izracunajFitness(self.spremnik))
        return najboljeRjesenje

def vizualizirajRjesenje(spremnik, jedinica):
    fig, ax = plt.subplots()
    ax.set_xlim(0, spremnik.sirina)
    ax.set_ylim(0, spremnik.sirina)
    ax.set_aspect('equal', adjustable='box')

    for baterija in jedinica.baterije:
        elipsa = patches.Ellipse((baterija.x, baterija.y), baterija.duzina, baterija.sirina, angle=baterija.rotacija,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(elipsa)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vizualizacija rješenja')
    plt.grid(True)
    plt.show()

spremnik = Spremnik(1_000_000)
baterije = [Baterija(40_000, 20_000, 0, 0, random.randint(0, 1_000_000)) for _ in range(39)] + \
           [Baterija(20_000, 10_000, 0, 0, random.randint(0, 1_000_000)) for _ in range(161)]
velicinaPopulacije = 100
brojGeneracija = 1000
stopaMutacije = 0.1
stopaKrizanja = 0.9

genetskiAlgoritam = GenetskiAlgoritam(velicinaPopulacije, brojGeneracija, stopaMutacije, stopaKrizanja, spremnik, baterije)
najboljeRjesenje = genetskiAlgoritam.pokreni()

print("Najbolje rješenje:", najboljeRjesenje)
with open('rjesenje.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for baterija in najboljeRjesenje.baterije:
        writer.writerow([int(baterija.x*1000), int(baterija.y*1000), baterija.duzina, baterija.sirina, int(baterija.rotacija*1000)])

vizualizirajRjesenje(spremnik, najboljeRjesenje)