#Cod. op. M
import tensorflow as tf

# Declarar las matrices
X = tf.constant([[-1, 3, 4],
                 [ 2, 6, 4],
                 [-3, 1, 5]], dtype=tf.float32)

Y = tf.constant([[ 2, -5, 4],
                 [ 1,  2, 6],
                 [ 3, -1, 5]], dtype=tf.float32)

# Multiplicación de matrices
W = tf.linalg.matmul(X, Y)

# Inversa y determinante de X
Z = tf.linalg.inv(X)
V = tf.linalg.det(X)

#Confirmar inversa
I = tf.linalg.matmul(X,Z)

#Matriz de ceros
C = tf.zeros([3,3])

#Matriz de unos
U = tf.ones([2,4])

# Mostrar resultados
print("Resultado de X * Y:\n", W.numpy())
print("\nInversa de X:\n", Z)
print("\nDeterminante de X:\n", V)
print("\nMatriz X*X^-1(Z):\n", I)
print("\nMatriz de ceros:\n", C)
print("\nMatriz de unos:\n", U)

