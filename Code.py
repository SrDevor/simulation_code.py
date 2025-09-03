# -*- coding: utf-8 -*-
"""

Simulación de Geometría Curva Emergente de Correlaciones Cuánticas

Autor: Sergio Martínez Sánchez
Fecha: 03-09-2025

Descripción:
Este script implementa la simulación numérica del modelo de 3 qubits presentado
en el paper "Fundamentos de la Geometría Emergente". Calcula cómo la curvatura
de un espacio discreto emergente depende de la asimetría en la estructura de
entrelazamiento del estado fundamental del sistema.

El flujo de trabajo es el siguiente:
1. Se define un rango para el parámetro de asimetría eta (J'/J).
2. Para cada valor de eta:
   a. Se construye el Hamiltoniano del subespacio M=-1/2.
   b. Se encuentra el estado fundamental mediante diagonalización exacta.
   c. Se calculan las informaciones mutuas bipartitas (I_ij).
   d. Se calculan las distancias emergentes (d_ij) según el postulado.
   e. Se calcula la curvatura (kappa) como el déficit angular normalizado.
3. Se visualiza la curvatura en función de eta, generando el gráfico principal
   de la investigación.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.style as style

#  PARÁMETROS GLOBALES DEL MODELO 
# Estos parámetros se basan directamente en la teoría desarrollada.

# Constante de escala de longitud en el postulado de la métrica.
# La fijamos a 1.0 sin pérdida de generalidad, ya que solo escala el tamaño
# general del triángulo emergente, no su forma ni su curvatura.
ALPHA = 1.0

# Exponente en el postulado de la métrica d = alpha / I^k.
# Como se discutió en la teoría, usamos k=0.5.
K = 0.5

# Escala de energía global J. Se fija a 1.0, ya que solo escala los
# autovalores de energía, no los autoestados.
J_COUPLING = 1.0

# Rango y resolución del barrido del parámetro de asimetría eta.
ETA_MIN = 0.1
ETA_MAX = 3.0
ETA_STEPS = 400

# Pequeña cons tante para evitar divisiones por cero o logaritmos de cero.
EPSILON = 1e-12

#  FUNCIONES AUXILIARES Y DE CÁLCULO 

def binary_entropy(p: float) -> float:
    """
    Calcula la entropía binaria H_b(p) de forma numéricamente estable.
    H_b(p) = -p*log2(p) - (1-p)*log2(1-p).

    Args:
        p (float): Probabilidad, debe estar en el rango [0, 1].

    Returns:
        float: El valor de la entropía binaria en bits.
    """
    if p < EPSILON or (1 - p) < EPSILON:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def setup_hamiltonian(eta: float) -> np.ndarray:
    """
    Construye la matriz Hamiltoniana 3x3 para el subespacio M=-1/2.
    La matriz se deriva directamente de la teoría (Apéndice A).

    Args:
        eta (float): Parámetro de asimetría J'/J.

    Returns:
        np.ndarray: La matriz Hamiltoniana 3x3.
    """
    H = np.array([
        [1 - 2*eta, 2,       2*eta    ],
        [2,         -1,      2        ],
        [2*eta,     2,       -1 - 2*eta]
    ])
    return J_COUPLING * H

def find_ground_state(H: np.ndarray) -> np.ndarray:
    """
    Diagonaliza el Hamiltoniano y devuelve el estado fundamental.
    Utiliza scipy.linalg.eigh, que es eficiente para matrices hermitianas
    y devuelve los autovalores ordenados.

    Args:
        H (np.ndarray): La matriz Hamiltoniana.

    Returns:
        np.ndarray: El vector de estado fundamental normalizado (c1, c2, c3).
    """
    eigenvalues, eigenvectors = la.eigh(H)
    # El primer autovector (columna 0) corresponde al autovalor más bajo.
    ground_state_vector = eigenvectors[:, 0]
    return ground_state_vector

def calculate_observables(ground_state_vector: np.ndarray) -> dict:
    """
    Calcula todas las cantidades físicas derivadas del estado fundamental.
    Sigue las derivaciones del Apéndice B de la teoría.

    Args:
        ground_state_vector (np.ndarray): Vector (c1, c2, c3).

    Returns:
        dict: Un diccionario conteniendo informaciones mutuas, distancias y curvatura.
    """
    c1, c2, c3 = ground_state_vector
    
    # 1. Calcular las probabilidades |c_i|^2
    p1 = np.abs(c1)**2
    p2 = np.abs(c2)**2
    p3 = np.abs(c3)**2
    
    # 2. Calcular las Informaciones Mutuas usando la entropía binaria
    # I(2:3) ~ |c1|^2, I(3:1) ~ |c2|^2, I(1:2) ~ |c3|^2
    I_23 = binary_entropy(p1)
    I_31 = binary_entropy(p2)
    I_12 = binary_entropy(p3)
    
    # 3. Calcular las distancias emergentes según el postulado
    # d = alpha / I^k. Se maneja el caso I=0 para evitar división por cero.
    d_12 = ALPHA / (I_12**K + EPSILON)
    d_23 = ALPHA / (I_23**K + EPSILON)
    d_31 = ALPHA / (I_31**K + EPSILON)
    
    # 4. Calcular la curvatura a partir del déficit angular
    # Se usan las longitudes de los lados a=d23, b=d31, c=d12
    a, b, c = d_23, d_31, d_12
    
    # Verificar la desigualdad triangular. Si no se cumple, la curvatura no
    # está bien definida en el sentido euclidiano.
    if not (a + b > c and a + c > b and b + c > a):
        # Este es un resultado físicamente interesante: el espacio "se rompe".
        # Asignamos NaN (Not a Number) para excluirlo del gráfico principal.
        kappa = np.nan
    else:
        try:
            # Ley de los cosenos para encontrar los ángulos
            cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_beta = (a**2 + c**2 - b**2) / (2 * a * c)
            cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
            
            # Clip para estabilidad numérica de arccos
            angle_alpha = np.arccos(np.clip(cos_alpha, -1, 1))
            angle_beta = np.arccos(np.clip(cos_beta, -1, 1))
            angle_gamma = np.arccos(np.clip(cos_gamma, -1, 1))
            
            angle_sum = angle_alpha + angle_beta + angle_gamma
            
            # Fórmula de Herón para el área del triángulo
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            
            # Déficit angular normalizado por el área
            kappa = (angle_sum - np.pi) / (area + EPSILON)
        except (ValueError, ZeroDivisionError):
            kappa = np.nan

    return {
        "mutual_info": (I_12, I_23, I_31),
        "distances": (d_12, d_23, d_31),
        "curvature": kappa
    }

#  BUCLE PRINCIPAL DE SIMULACIÓN 

def run_simulation():
    """
    Ejecuta el barrido completo sobre el parámetro eta y recopila los resultados.
    """
    print("Iniciando simulación del modelo de geometría emergente...")
    
    eta_values = np.linspace(ETA_MIN, ETA_MAX, ETA_STEPS)
    kappa_values = []
    
    for i, eta in enumerate(eta_values):
        # Imprimir progreso
        if (i + 1) % (ETA_STEPS // 10) == 0:
            print(f"Progreso: {100 * (i + 1) / ETA_STEPS:.0f}%")

        # Secuencia de cálculo para cada eta
        H = setup_hamiltonian(eta)
        gs_vector = find_ground_state(H)
        observables = calculate_observables(gs_vector)
        
        kappa_values.append(observables["curvature"])
        
    print("Simulación completada.")
    return eta_values, np.array(kappa_values)

#  VISUALIZACIÓN DE RESULTADOS 

def plot_results(eta_values, kappa_values):
    """
    Genera el gráfico principal: Curvatura vs. Asimetría del Entrelazamiento.
    """
    print("Generando gráfico de resultados...")

    # Estilo de la gráfica para una apariencia profesional
    style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    # Graficar los datos principales
    print("Min kappa:", np.nanmin(kappa_values))
    print("Max kappa:", np.nanmax(kappa_values))

    plt.plot(eta_values, kappa_values, label='Curvatura Emergente $\\kappa$', color='darkblue', linewidth=2.5)
    
    # Líneas de referencia importantes
    # Geometría plana (Euclidiana)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.2, label='Espacio Plano ($\\kappa=0$)')
    # Punto de máxima simetría
    plt.axvline(1.0, color='red', linestyle=':', linewidth=1.5, label='Punto Simétrico ($\\eta=1$)')
    
    # Anotaciones para una mejor interpretación
    plt.annotate(
        'Geometría Plana', 
        xy=(1.0, 0), 
        xytext=(1.2, 0.5),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8)
    )
    
    plt.annotate(
        'Régimen Hiperbólico\n(Curvatura Negativa)',
        xy=(-0.2, np.nanmin(kappa_values) * 3),
        ha='center',
        fontsize=12,
        color='darkorange'
    )
    
    plt.annotate(
        'Régimen Esférico\n(Curvatura Positiva)',
        xy=(2.0, np.nanmax(kappa_values) / 2),
        ha='center',
        fontsize=12,
        color='darkgreen'
    )
    
    # Títulos y etiquetas con formato LaTeX para rigor matemático
    plt.title('Espacio de Fases de Geometrías Emergentes', fontsize=18, fontweight='bold')
    plt.xlabel(r"Parámetro de Asimetría del Entrelazamiento, $\eta = J'/J$", fontsize=14)
    plt.ylabel(r'Curvatura Discreta Emergente, $\kappa$', fontsize=14)

    # Límites y escala de los ejes
    plt.xlim(ETA_MIN, ETA_MAX)
    # Ajustar el límite y para mejor visualización
    max_abs_kappa = np.nanmax(np.abs(kappa_values))
    plt.ylim(-max_abs_kappa*1.1, max_abs_kappa*1.1)
    
    # Añadir leyenda y mejorar la apariencia
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    #plt.tight_layout()
    
    # Guardar la figura en alta resolución
    output_filename = 'curvatura_emergente_vs_eta.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Gráfico guardado como '{output_filename}'")
    
    # Mostrar la figura
    plt.show()

#  PUNTO DE ENTRADA DEL SCRIPT 
if __name__ == '__main__':
    eta_data, kappa_data = run_simulation()
    plot_results(eta_data, kappa_data)
