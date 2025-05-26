import React, { useEffect, useState, useRef, memo } from "react";
import { DeckGL } from "deck.gl";
import { Map } from "react-map-gl/maplibre";
import { IconLayer, ScatterplotLayer, PathLayer } from "@deck.gl/layers";

// Usa un mapa gratuito de MapLibre en lugar de requerir API key
const MAP_STYLE = `https://api.maptiler.com/maps/streets/style.json?key=MZjUAQpw10B8E0nsKVQP`;

const INITIAL_VIEW_STATE = {
  latitude: 23.1136,
  longitude: -82.3666,
  zoom: 13,
  bearing: 0,
  pitch: 0,
};

// Componente de formulario optimizado con React.memo
const OptimizationForm = memo(({ 
  optimizationParams, 
  setOptimizationParams, 
  requestOptimization, 
  setOptimizedRoutes 
}) => {
  // Estado local para los valores de entrada
  const [localStartPoint, setLocalStartPoint] = useState(optimizationParams.start_point);
  const [localTargetPoints, setLocalTargetPoints] = useState(
    optimizationParams.target_points.join(',')
  );
  const [localNumTrucks, setLocalNumTrucks] = useState(optimizationParams.num_trucks);
  
  // Manejar la sincronización del estado local cuando cambian los props
  useEffect(() => {
    setLocalStartPoint(optimizationParams.start_point);
    setLocalTargetPoints(optimizationParams.target_points.join(','));
    setLocalNumTrucks(optimizationParams.num_trucks);
  }, [optimizationParams]);
  
  // Función para aplicar los cambios al estado principal
  const applyChanges = () => {
    setOptimizationParams({
      ...optimizationParams,
      start_point: localStartPoint,
      target_points: localTargetPoints.split(',').map(p => p.trim()).filter(p => p),
      num_trucks: parseInt(localNumTrucks)
    });
  };
  
  return (
    <div style={{
      position: "absolute",
      top: "10px",
      right: "10px",
      zIndex: 100,
      padding: "10px",
      backgroundColor: "rgba(255,255,255,0.8)",
      borderRadius: "4px",
      maxWidth: "300px"
    }}>
      <h3>Optimización de Rutas</h3>
      
      <div>
        <label>
          Punto de inicio:
          <input 
            type="text" 
            value={localStartPoint} 
            onChange={e => setLocalStartPoint(e.target.value)}
          />
        </label>
      </div>
      
      <div>
        <label>
          Puntos objetivo (separados por coma):
          <input 
            type="text" 
            value={localTargetPoints} 
            onChange={e => setLocalTargetPoints(e.target.value)}
          />
        </label>
      </div>
      
      <div>
        <label>
          Número de camiones:
          <input 
            type="number" 
            value={localNumTrucks} 
            onChange={e => setLocalNumTrucks(e.target.value)}
          />
        </label>
      </div>
      
      <button onClick={() => {
        applyChanges();
        requestOptimization();
      }}>
        Optimizar Rutas
      </button>
      
      <button onClick={() => setOptimizedRoutes([])}>
        Limpiar Rutas
      </button>
    </div>
  );
});

function App() {
  const [vehicleData, setVehicleData] = useState({});
  const [trafficLights, setTrafficLights] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState("Conectando...");
  const [errorMessage, setErrorMessage] = useState("");
  const [optimizedRoutes, setOptimizedRoutes] = useState([]);
  const [showOptimizationForm, setShowOptimizationForm] = useState(false);
  const [optimizationParams, setOptimizationParams] = useState({
    start_point: "",
    target_points: [],
    num_trucks: 3,
    truck_capacities: [10, 10, 10],
    target_demands: {}
  });
  
  const trailsRef = useRef({});
  const wsRef = useRef(null);

  useEffect(() => {
    // Manejo mejorado de la conexión WebSocket
    const connectWebSocket = () => {
      try {
        console.log("Intentando conectar al WebSocket...");
        const ws = new WebSocket("ws://localhost:8765");
        wsRef.current = ws;

        ws.onopen = () => {
          console.log("Conexión WebSocket establecida");
          setConnectionStatus("Conectado");
          setErrorMessage("");
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // Si es una actualización de posiciones
            if (data.vehicles) {
              console.log(`Recibidos datos para ${data.vehicles.length} vehículos`);
              
              const updated = {};
              data.vehicles.forEach((v) => {
                const prevTrail = trailsRef.current[v.id] || [];
                const newTrail = [...prevTrail.slice(-20), [v.lon, v.lat]];
                trailsRef.current[v.id] = newTrail;

                updated[v.id] = {
                  id: v.id,
                  position: [v.lon, v.lat],
                  trail: newTrail,
                  color: 0,
                };
              });

              setVehicleData(updated);
              setTrafficLights(data.traffic_lights || []);
            }
            
            // Si es una respuesta de optimización
            if (data.type === 'optimization_result') {
              console.log("Recibidos resultados de optimización:", data);
              
              // Transformar las rutas al formato esperado por PathLayer
              const routes = data.routes.map((route, idx) => {
                return {
                  id: `route-${idx}`,
                  path: route.map(point => [point.lon, point.lat]),
                  color: getRouteColor(idx)
                };
              });
              
              setOptimizedRoutes(routes);
            }
            
            // Si hay un error de optimización
            if (data.type === 'optimization_error') {
              setErrorMessage(data.message);
            }
            
          } catch (err) {
            console.error("Error procesando mensaje:", err);
            setErrorMessage(`Error procesando datos: ${err.message}`);
          }
        };

        ws.onclose = () => {
          console.log("Conexión WebSocket cerrada");
          setConnectionStatus("Desconectado");
          // Reintentar conexión después de 3 segundos
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error("Error en WebSocket:", error);
          setConnectionStatus("Error");
          setErrorMessage("Error en la conexión WebSocket");
        };
      } catch (err) {
        console.error("Error creando WebSocket:", err);
        setConnectionStatus("Error");
        setErrorMessage(`Error de conexión: ${err.message}`);
        // Reintentar conexión después de 3 segundos
        setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Función para enviar solicitud de optimización
  const requestOptimization = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'optimization_request',
        ...optimizationParams
      }));
    } else {
      setErrorMessage("No hay conexión con el servidor");
    }
  };
  
  // Función para generar colores de ruta distintos
  const getRouteColor = (index) => {
    const colors = [
      [255, 0, 0],    // Rojo
      [0, 128, 255],  // Azul
      [0, 204, 0],    // Verde
      [255, 102, 0],  // Naranja
      [153, 51, 255], // Púrpura
      [255, 204, 0],  // Amarillo
      [0, 204, 204],  // Turquesa
      [255, 0, 255],  // Magenta
    ];
    
    return colors[index % colors.length];
  };

  const layers = [
    // Capa para rutas optimizadas
    new PathLayer({
      id: 'optimized-routes',
      data: optimizedRoutes,
      pickable: true,
      widthScale: 10,
      widthMinPixels: 2,
      getPath: d => d.path,
      getColor: d => d.color,
      getWidth: 5
    }),
    
    // Capas existentes
    new IconLayer({
      id: "vehicle-icons",
      data: Object.values(vehicleData),
      getIcon: (d) => ({
        url: "/icons/car.png",
        width: 128,
        height: 128,
        anchorY: 128,
      }),
      getPosition: (d) => d.position,
      getSize: 2, // escala del ícono
      sizeScale: 10,
      getAngle: 0,
      getColor: 0,
      pickable: true,
    }),

    new ScatterplotLayer({
      id: "traffic-lights",
      data: trafficLights,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: (d) => d.state === "red" ? [255, 0, 0] : [0, 255, 0],
      getRadius: 10,
      radiusMinPixels: 2,
    }),
  ];
  
  return (
    <>
      {/* Panel de estado existente */}
      <div
        style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          zIndex: 100,
          padding: "10px",
          backgroundColor: "rgba(255,255,255,0.7)",
          borderRadius: "4px",
          maxWidth: "300px",
        }}
      >
        <div>
          <strong>Estado:</strong> {connectionStatus}
        </div>
        {errorMessage && (
          <div style={{ color: "red" }}>{errorMessage}</div>
        )}
        <div>
          <strong>Vehículos:</strong> {Object.keys(vehicleData).length}
        </div>
        <button onClick={() => setShowOptimizationForm(!showOptimizationForm)}>
          {showOptimizationForm ? "Ocultar Optimización" : "Mostrar Optimización"}
        </button>
      </div>
      
      {/* Formulario de optimización */}
      {showOptimizationForm && 
        <OptimizationForm 
          optimizationParams={optimizationParams}
          setOptimizationParams={setOptimizationParams}
          requestOptimization={requestOptimization}
          setOptimizedRoutes={setOptimizedRoutes}
        />
      }
      
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
      >
        <Map reuseMaps mapLib={import("maplibre-gl")} mapStyle={MAP_STYLE} />
      </DeckGL>
    </>
  );
}

export default App;
