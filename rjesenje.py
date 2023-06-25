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
            for j, drugaBaterija in enumerate(self.baterije[i + 1:]):
                if self.preklapajuSe(baterija, drugaBaterija):
                    return 0
        return ukupnaPovrsina

    @staticmethod
    def preklapajuSe(b1, b2):
        tockeNaB1 = [(np.cos(t) * b1.duzina, np.sin(t) * b1.sirina) for t in np.linspace(0, 2 * np.pi, 100)]
        rotiraneTockeB1 = [Jedinica.rotirajTocku(tocka, np.radians(b1.rotacija)) for tocka in tockeNaB1]
        rotiraneTockeB1 = [(x + b1.x, y + b1.y) for x, y in rotiraneTockeB1]
        
        tockeNaB2 = [(np.cos(t) * b2.duzina, np.sin(t) * b2.sirina) for t in np.linspace(0, 2 * np.pi, 100)]
        rotiraneTockeB2 = [Jedinica.rotirajTocku(tocka, np.radians(b2.rotacija)) for tocka in tockeNaB2]
        rotiraneTockeB2 = [(x + b2.x, y + b2.y) for x, y in rotiraneTockeB2]
        
        osi = Jedinica.normalniVektoriElipse(b1.duzina, b1.sirina, np.radians(b1.rotacija)) + \
              Jedinica.normalniVektoriElipse(b2.duzina, b2.sirina, np.radians(b2.rotacija))

        for os in osi:
            proj1 = Jedinica.projekcijaTocakaNaOs(rotiraneTockeB1, os)
            proj2 = Jedinica.projekcijaTocakaNaOs(rotiraneTockeB2, os)
            if not Jedinica.intervalPreklapanja(proj1, proj2):
                return False
        return True

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
            baterija.x = random.uniform(0, spremnik.sirina - baterija.duzina)
            baterija.y = random.uniform(0, spremnik.sirina - baterija.sirina)
            baterija.rotacija = random.randint(0, 179999) / 1000.0
            rotiraneTocke = [Jedinica.rotirajTocku(tocka, np.radians(baterija.rotacija)) for tocka in tockeNaElipsi]
            rotiraneTocke = [(x + baterija.x, y + baterija.y) for x, y in rotiraneTocke]
            preklapanje = False

            for drugaBaterija in self.baterije:
                if drugaBaterija != baterija:
                    tockeNaDrugojElipsi = [(np.cos(t) * drugaBaterija.duzina, np.sin(t) * drugaBaterija.sirina) for t in np.linspace(0, 2 * np.pi, 100)]
                    rotiraneTockeDrugeBaterije = [Jedinica.rotirajTocku(tocka, np.radians(drugaBaterija.rotacija)) for tocka in tockeNaDrugojElipsi]
                    rotiraneTockeDrugeBaterije = [(x + drugaBaterija.x, y + drugaBaterija.y) for x, y in rotiraneTockeDrugeBaterije]
                    osi = Jedinica.normalniVektoriElipse(baterija.duzina, baterija.sirina, np.radians(baterija.rotacija)) + \
                          Jedinica.normalniVektoriElipse(drugaBaterija.duzina, drugaBaterija.sirina, np.radians(drugaBaterija.rotacija))
                    if all(Jedinica.intervalPreklapanja(Jedinica.projekcijaTocakaNaOs(rotiraneTocke, os), 
                                                        Jedinica.projekcijaTocakaNaOs(rotiraneTockeDrugeBaterije, os)) for os in osi):
                        preklapanje = True
                        break

            brojPokusaja += 1

            if not preklapanje or brojPokusaja > 100:
                break

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
                x = random.uniform(0, maksX)
                y = random.uniform(0, maksY)
                jedinicaBaterije.append(Baterija(b.duzina, b.sirina, x, y, b.rotacija))
            populacija.append(Jedinica(jedinicaBaterije))
        return populacija

    def pokreni(self):
        for generacija in range(self.brojGeneracija):
            dobrote = [jedinica.izracunajFitness(self.spremnik) for jedinica in self.populacija]
            if sum(dobrote) == 0:
                indeksi = range(len(self.populacija))
                izabraniIndeksi = random.sample(indeksi, 2)
                roditelj1 = self.populacija[izabraniIndeksi[0]]
                roditelj2 = self.populacija[izabraniIndeksi[1]]
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
        centarX = baterija.x + baterija.duzina / 2.0
        centarY = baterija.y + baterija.sirina / 2.0
        elipsa = patches.Ellipse((centarX, centarY), baterija.duzina, baterija.sirina, angle=baterija.rotacija, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(elipsa)
    plt.xlim(0, spremnik.sirina)
    plt.ylim(0, spremnik.sirina)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vizualizacija rasporeda baterija")
    plt.show()

if __name__ == "__main__":
    spremnik = Spremnik(1_000_000)
    baterije = [Baterija(40_000, 20_000, 0, 0, random.randint(0, 1_000_000)) for _ in range(39)] + \
               [Baterija(20_000, 10_000, 0, 0, random.randint(0, 1_000_000)) for _ in range(161)]
    velicinaPopulacije = 100
    brojGeneracija = 1000
    stopaMutacije = 0.1
    stopaKrizanja = 0.8
    genetskiAlgoritam = GenetskiAlgoritam(velicinaPopulacije, brojGeneracija, stopaMutacije, stopaKrizanja, spremnik, baterije)
    najboljeRjesenje = genetskiAlgoritam.pokreni()

    print("Najbolje rješenje:", najboljeRjesenje)
    with open('rjesenje.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for baterija in najboljeRjesenje.baterije:
            writer.writerow([int(baterija.x), int(baterija.y), baterija.duzina, baterija.sirina, int(baterija.rotacija)])

    vizualizirajRjesenje(spremnik, najboljeRjesenje)