 
mithisDir = fullfile('C:\Users\Pedro\Documents\TRABALLO_CASA\08_HFRPROGS\hfrprogs-master\matlab', 'mi_demo')
mi_mode_min_scale = 20; % 5 km
mi_modes_fn = fullfile( mithisDir, 'misdatos', 'modes.mat' );

% This border was previously created by taking a California coastline,
% cutting a piece out, then interpolating that piece so that no edge
% segment is short than about 0.3 km and then adding to the resulting
% smoothed coastline the open boundary part of the boundary.
mi_border = load( fullfile( mithisDir, 'misdatos', 'borde_galicia_2.csv' ) );

% These would typically be initially determined by running
% generate_OMA_modes with the keyboard input argument set to true and
% examining the numbers assigned to the edges of the domain by PDETOOL in
% Boundary Mode with "Show Edge Labels".
 mi_ob_nums = { 40:85 };
 mi_db_nums = { 23 };
%mi_ob_nums = { 64:68 };
%mi_db_nums = { 30 };
LonLims = [-11.178,-8.236];
LatLims = [40.695,44.637];
m_proj('mercator','lon',LonLims + [-0.2,0.2] * diff(LonLims), ...
         'lat',LatLims + [-0.2,0.2] * diff(LatLims));
% Generate modes
generate_OMA_modes( mi_modes_fn, mi_border, mi_mode_min_scale, mi_ob_nums, mi_db_nums, ...
                      [], [], false );