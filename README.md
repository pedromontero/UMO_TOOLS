# UMO_TOOLS

Diferentes ferramentas en Python usadas pola ***UMO- INTECMAR***

**drawmap:** Programa para ler un ficheiro MOHID HDF5 ou NetCDF e
debuxar un mapa. Usa *drawcurrents.py* que deberá ir para *commons* 
cando estea máis estável.

**common** Modulo incluindo:

* *readers:* Conxunto de clases lectoras de arquivos HDF e NetCDF: 
*ReaderHDF* e *ReaderNCDF*
* *inout:* función *input* para ler un ficheiro json coas opcións
de cada programa
* *boundary_box:* Clase para establecer os limites dun mapa con
algunhas operación sinxelas