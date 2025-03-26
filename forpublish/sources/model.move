module tensorflowsui::model {
    use sui::tx_context::TxContext;
    use tensorflowsui::graph;

    public fun create_model_signed_fixed(graph: &mut graph::SignedFixedGraph, scale: u64) {


    }

    public entry fun initialize(ctx: &mut TxContext) {
        let mut graph = graph::create_signed_graph(ctx);
        create_model_signed_fixed(&mut graph, 2); 
        let mut partials = graph::create_partial_denses(ctx);
        graph::add_partials_for_all_but_last(&graph, &mut partials);
        graph::share_graph(graph);
        graph::share_partial(partials);
    }
}