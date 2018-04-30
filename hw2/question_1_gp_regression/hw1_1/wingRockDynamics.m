function pDot = wingRockDynamics(p, phi)
    pDot = .8 + .2314*phi + .6918*p - .6245*abs(phi)*p + .0095*abs(p)*p + .0214*phi^3;
end


