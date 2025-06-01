import React from 'react';

const RouteInfo = ({ routes, onSelectRoute }) => (
  <div style={{
    position: 'absolute',
    bottom: '20px',
    right: '20px',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: '15px',
    borderRadius: '8px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
    maxWidth: '320px',
    maxHeight: '40%',
    overflowY: 'auto',
    zIndex: 100
  }}>
    <h3 style={{ margin: '0 0 10px 0', fontSize: '16px', fontWeight: 'bold' }}>
      Rutas Optimizadas
    </h3>
    
    <div>
      {routes.map((route, index) => (
        <div 
          key={route.id}
          style={{
            padding: '8px',
            margin: '5px 0',
            borderRadius: '4px',
            border: '1px solid #e0e0e0',
            backgroundColor: route.selected ? '#f0f0f0' : 'white',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center'
          }}
          onClick={() => onSelectRoute(route.id)}
        >
          <div 
            style={{ 
              width: '12px', 
              height: '12px', 
              borderRadius: '50%', 
              backgroundColor: `rgb(${route.color.join(',')})`,
              marginRight: '8px'
            }} 
          />
          <div>
            <div style={{ fontWeight: route.selected ? 'bold' : 'normal' }}>
              Ruta {index + 1}
            </div>
            {route.distance && (
              <div style={{ fontSize: '12px', color: '#666' }}>
                Distancia: {route.distance.toFixed(2)} km
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default RouteInfo;