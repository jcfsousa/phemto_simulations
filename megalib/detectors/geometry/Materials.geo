// This file contains materials used by all geometries in alphabetical order

AbsorptionFileDirectory $(MEGALIB)/resource/examples/geomega/materials

Material BGO
BGO.Density					7.1
BGO.Component				Bi	4  
BGO.Component				Ge	3   
BGO.Component				O	12   

Material CircuitBoard
CircuitBoard.Density            1.8
CircuitBoard.ComponentByMass    H   0.030  
CircuitBoard.ComponentByMass    C   0.174  
CircuitBoard.ComponentByMass    O   0.392  
CircuitBoard.ComponentByMass    Al   0.100  
CircuitBoard.ComponentByMass    Si   0.244  
CircuitBoard.ComponentByMass    Fe   0.010  
CircuitBoard.ComponentByMass    Cu   0.030  
CircuitBoard.ComponentByMass    Sn   0.010  
CircuitBoard.ComponentByMass    Pb   0.010

Material Copper
Copper.Density				8.954
Copper.Component			Cu	1

Material CsI
CsI.Density					4.5
CsI.Component				Cs	1
CsI.Component				I	1
// Change CsI to tungsten; testing purposes
#CsI.Density				19.3
#CsI.Component				W	1

Material CZT
CZT.Density					5.78 // jose: Changed according to Aline thesis
CZT.ComponentByAtoms		Cd	9
CZT.ComponentByAtoms		Zn	1
CZT.ComponentByAtoms		Te	10

Material CdTe 
CdTe.Density					5.85 // 
CdTe.ComponentByAtoms		Cd	1
CdTe.ComponentByAtoms		Te	1

// Plastic scintillator, estimated from online values for EJ-200 plastic scintillator
// These values and this material selection needs to be checked!
Material PlasticScin
PlasticScin.Density				1.023
PlasticScin.ComponentByMass		C	0.48
PlasticScin.ComponentByMass		H	0.52

Material PCB
PCB.Density					1.2
PCB.Component				H   8
PCB.Component				C   5
PCB.Component				O   2

// Sandwich (honeycomb) materials
// Toray M55J/RS3C carbon fiber (honeycomb face-sheet)
Material CarbonFiber
CarbonFiber.Density			1.9
CarbonFiber.Component		C	1

// HRH-10 Nomex fiber (honeycomb core)
Material HC_core
HC_core.Density				0.08
HC_core.Component			C	4
HC_core.Component			N	4
HC_core.Component			H	4
HC_core.Component			O	4

Material Silicon
Silicon.Density				2.33
Silicon.Component			Si	1

Material Tungsten
Tungsten.Density			19.25
Tungsten.ComponentByMass	W  1

Material Vacuum
Vacuum.Density				1E-12
Vacuum.Component			H   1

