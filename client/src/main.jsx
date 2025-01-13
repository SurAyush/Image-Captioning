import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider, createBrowserRouter } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout'
import Home from './pages/Home'
import Caption from './pages/Caption'
import Error from './pages/Error'


const router = createBrowserRouter([
  {
    path: '/',
    element:  <Layout />,
    errorElement: <Error />,
    children:[
      {index: true, element: <Home />},
      {path:"caption", element: <Caption />},
    ]
  }
]);

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router}/>
  </StrictMode>,
)
