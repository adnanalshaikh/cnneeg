
function [ signals, gmin, gmax ] = load_signals ( datapath )

% 100 file for each group, 4097 sample in each file
% The last column is a label to the group (A -> 1, B->2, C->3, D->4, E->5)
num_files      = 100;
num_samples    = 4097;
group_folders  = ['Z', 'O', 'N', 'F', 'S'];
groups         = ['A', 'B', 'C', 'D', 'E'];

signals = zeros(num_files, num_samples + 2);

% Load all signals and return the maximum and minimum reading 
gmax = -realmax;
gmin =  realmax;

for i=1:length(group_folders)
    gfolderpath = [datapath, group_folders(i), '\'];
    
    for j = 1 : num_files
        filepath = sprintf( '%s%c%03d.txt', gfolderpath, group_folders(i), j);
        d = load(filepath,'-ascii');
        
        maxd = max(d);
        mind = min(d);
        
        if ( gmax < maxd ) 
            gmax = maxd;
        end
        if ( gmin > mind ) 
            gmin = mind;
        end
         
        signals((i-1)*100+j, 1:4097)  = d;
        signals((i-1)*100+j,   4098)  = i;
        signals((i-1)*100+j,   4099)  = j;
     end 
end

end

