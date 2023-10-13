# Importa la clase flabianos desde el archivo listaPermitidos.py
from listaPermitidos import flabianos

# Crea una instancia de la clase flabianos
invitados = flabianos()

# Nombre de la persona que deseas verificar
nombre_a_verificar = 'JOSUE'

# Llama al método TuSiTuNo para verificar si la persona está en la lista de invitados
invitados.TuSiTuNo(nombre_a_verificar)