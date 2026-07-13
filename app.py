import streamlit as st
import cv2
import numpy as np
import pandas as pd


# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Pro-Image Lab",
    layout="wide"
)

st.title("🎛️ Pro-Image Lab")
st.subheader("Detección y análisis de bloques")


# =====================================================
# FUNCIONES
# =====================================================

def cargar_imagen(archivo):

    datos = np.asarray(
        bytearray(archivo.read()),
        dtype=np.uint8
    )

    imagen = cv2.imdecode(
        datos,
        cv2.IMREAD_COLOR
    )

    imagen = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2RGB
    )

    return imagen



def modificar_rgb(imagen, r, g, b):

    img = imagen.astype(np.float32)

    img[:,:,0] *= r
    img[:,:,1] *= g
    img[:,:,2] *= b

    img = np.clip(
        img,
        0,
        255
    )

    return img.astype(np.uint8)



def modificar_gris(gris, brillo):

    img = gris.astype(np.float32)

    img *= brillo

    img = np.clip(
        img,
        0,
        255
    )

    return img.astype(np.uint8)



def umbralizar(gris, metodo, valor):

    if metodo == "Binario":

        _, salida = cv2.threshold(
            gris,
            valor,
            255,
            cv2.THRESH_BINARY
        )


    elif metodo == "Binario Invertido":

        _, salida = cv2.threshold(
            gris,
            valor,
            255,
            cv2.THRESH_BINARY_INV
        )


    else:

        _, salida = cv2.threshold(
            gris,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )


    return salida



def detectar_bloques(imagen, binaria, area_minima):


    # Buscar contornos
    contornos, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    resultado = imagen.copy()

    datos = []


    for contorno in contornos:


        area = cv2.contourArea(contorno)


        # eliminar ruido
        if area < area_minima:
            continue


        x,y,w,h = cv2.boundingRect(
            contorno
        )


        datos.append(
            {
                "x":x,
                "y":y,
                "Ancho (px)":w,
                "Alto (px)":h,
                "Área (px²)":int(area)
            }
        )


    # ordenar de izquierda a derecha usando el centro del objeto
    # (más confiable que la esquina superior para objetos rotados)
    datos = sorted(
        datos,
        key=lambda d: (d["x"] + d["Ancho (px)"]/2)
    )


    tabla=[]


    for i,d in enumerate(datos):


        x=d["x"]
        y=d["y"]
        w=d["Ancho (px)"]
        h=d["Alto (px)"]


        # Dibujar rectángulo

        cv2.rectangle(
            resultado,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            3
        )


        # Número del bloque

        cv2.putText(
            resultado,
            f"#{i+1}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )


        tabla.append(
            {
                "ID":i+1,
                "Ancho (px)":w,
                "Alto (px)":h,
                "Área (px²)":d["Área (px²)"]
            }
        )


    return resultado, tabla



# =====================================================
# CARGA DE IMAGEN
# =====================================================

archivo = st.sidebar.file_uploader(
    "📂 Cargar imagen",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if archivo:


    imagen = cargar_imagen(
        archivo
    )


    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_RGB2GRAY
    )


    # =================================================
    # EDICIÓN
    # =================================================

    st.header("🎨 Ajuste de imagen")


    col1,col2 = st.columns(2)



    with col1:

        st.subheader("RGB")


        r = st.slider(
            "Rojo",
            0.0,
            2.55,
            1.0,
            0.01
        )

        g = st.slider(
            "Verde",
            0.0,
            2.55,
            1.0,
            0.01
        )


        b = st.slider(
            "Azul",
            0.0,
            2.55,
            1.0,
            0.01
        )


        rgb = modificar_rgb(
            imagen,
            r,
            g,
            b
        )


        st.image(
            rgb,
            use_container_width=True
        )



    with col2:


        st.subheader("Escala de grises")


        brillo = st.slider(
            "Brillo",
            0.0,
            2.55,
            1.0,
            0.01
        )


        gris_final = modificar_gris(
            gris,
            brillo
        )


        st.image(
            gris_final,
            use_container_width=True
        )



    # =================================================
    # DETECCIÓN
    # =================================================


    st.divider()

    st.header("⬛⬜ Detección de bloques")


    c1,c2 = st.columns([1,2])


    with c1:


        metodo = st.selectbox(
            "Método",
            [
                "Binario",
                "Binario Invertido",
                "Otsu"
            ]
        )


        valor = st.slider(
            "Umbral",
            0,
            255,
            127
        )


        area_min = st.slider(
            "Área mínima",
            100,
            30000,
            1000
        )



    binaria = umbralizar(
        gris,
        metodo,
        valor
    )


    # Corrección para objetos oscuros
    if metodo=="Binario":

        binaria = cv2.bitwise_not(
            binaria
        )


    # NUEVO: para MOSTRAR usamos la binaria invertida
    # (fondo blanco, objeto negro) — esto es solo visual,
    # la detección de contornos sigue usando "binaria" tal cual
    binaria_display = cv2.bitwise_not(
        binaria
    )

    # convertir a 3 canales para poder dibujar los rectángulos rojos
    binaria_color = cv2.cvtColor(
        binaria_display,
        cv2.COLOR_GRAY2RGB
    )


    fondo = binaria_color


    imagen_detectada, tabla = detectar_bloques(
        fondo,
        binaria,
        area_min
    )



    with c2:

        st.image(
            imagen_detectada,
            caption="Bloques detectados",
            use_container_width=True
        )



    # =================================================
    # RESULTADOS
    # =================================================

    st.success(
        f"Bloques encontrados: {len(tabla)}"
    )


    if tabla:

        df = pd.DataFrame(
            tabla
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "No se detectaron bloques. Ajusta el umbral o el área mínima."
        )


else:

    st.info(
        "⬅️ Carga una imagen desde el menú lateral"
    )