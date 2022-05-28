function [ output_args ] = mark_alpha(alpha, opts)
%marks alpha with a dashed line on the current graph
DEF.col='k';
DEF.is_vert=true;
DEF.max_val=Inf;
DEF.lStyle='--';
DEF.lWidth=0.5;
if nargin < 2
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end
if opts.is_vert
    %plot([alpha,alpha],min(get(gca,'YLim'),opts.max_val),'LineStyle',opts.lStyle,...
        'LineWidth',opts.lWidth,'Color',opts.col);
else
    %plot(min(get(gca,'XLim'),opts.max_val),[alpha,alpha],'LineStyle',opts.lStyle,...
        'LineWidth',opts.lWidth,'Color',opts.col);
end
end

