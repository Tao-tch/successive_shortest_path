import pandas as pd
import igraph as ig
from collections import namedtuple

input_path = 'input.csv'
node = namedtuple('Node', ['ID', 'e_i', 'd_i'])
arc = namedtuple('Arc', ['Tail', 'Head', 'Capacity', 'Residual', 'Sent'])


def build_graph(inputs):
    fs_nodes, fs_arcs = dict(), dict()
    arcs = pd.DataFrame(columns=['tail', 'head', 'reversed', 'cost', 'capacity', 'sent'])
    node_df = pd.DataFrame(columns=['ID', 'in', 'out'])
    node_df.set_index('ID', inplace=True)
    sources, destinations = [], []
    for i, v in inputs.iterrows():
        tail, head, cost, capacity, flow = str(v['tail']), str(v['head']), int(v['cost']), int(v['capacity']), v['flow']
        if flow != 'none':
            flow = int(flow)
        else:
            flow = 0
        if flow > 0:
            sources.append([str(i+1), flow])
        if flow < 0:
            destinations.append([str(i+1), abs(flow)])
        if tail not in fs_nodes:
            fs_nodes[tail] = node(tail, [0], [0])
            node_df.loc[tail] = [0, 0]
        if head not in fs_nodes:
            fs_nodes[head] = node(head, [0], [0])
            node_df.loc[head] = [0, 0]
        fs_arcs[tail + '_' + head] = arc(tail, head, capacity, [0], [0])
        arcs.loc[len(arcs.index)] = [tail, head, False, cost, capacity, 0]
        arcs.loc[len(arcs.index)] = [head, tail, True, 2**31, capacity, capacity]
    for i in sources:
        for j in destinations:
            if i[0] + '_' + j[0] not in fs_arcs:
                fs_arcs[i[0] + '_' + j[0]] = arc(i[0], j[0], 2**31, [0], [0])
                arcs.loc[len(arcs.index)] = [i[0], j[0], False, 2**31, 2**31, 0]
                arcs.loc[len(arcs.index)] = [j[0], i[0], True, 2**31, 2**31, 2**31]
    fs_df = pd.DataFrame([[i[0], 0] for i in fs_nodes], columns=['nodes', 'count'])
    fs_df.set_index('nodes', inplace=True)
    for i in fs_arcs:
        fs_df.loc[fs_arcs[i].Tail] += 1
    points = [1]
    for i, r in fs_df.iterrows():
        points.append(points[-1] + r['count'])
    fs_df.loc[len(fs_df) + 1] = [0]
    fs_df['points'] = points
    fs_df.drop(columns=['count'], inplace=True)
    print(f'\n\nForwards Star:\n{fs_df}\n\n')
    return arcs, node_df, sources, destinations


def succesive_shortest_path(arcs, nodes, inflow, outflow):
    inflow_nodes = [i[0] for i in inflow]
    outflow_nodes = [i[0] for i in outflow]
    total = sum([i[1] for i in inflow])
    for i, _ in enumerate(inflow):
        nodes[f'{i}_pi'] = [0 for _ in range(len(nodes.index))]
    for i, _ in enumerate(outflow):
        arcs[f'{i}_pi'] = arcs['cost']
    sent, x = 0, 1
    print('Algorithm Start:\n')
    while True:
        print(f'iteration: {x}')
        for y, z in enumerate(inflow):

            # 1. Residual Network
            df = arcs.loc[arcs['capacity'] - arcs['sent'] > 0, ['tail', 'head', f'{y}_pi', 'capacity']]
            graph = ig.Graph.TupleList(df.itertuples(index=False), directed=True, edge_attrs=[f'{y}_pi'])
            print('(1) residual network')
            for i in df.itertuples(index=False):
                print(f'   {i.tail},{i.head}:({i[3]},{i.capacity})')

            # 2. Shortest Path
            distances = graph.distances(source=z[0], weights=graph.es[f'{y}_pi'], mode='out')[0]
            print('(2) shortest path')
            for i, v in zip(nodes.index, distances):
                print(f'   {i}: {int(v)}')

            # 3. node potential
            nodes[f'{y}_pi'] = nodes[f'{y}_pi'] - distances  # update the key later on
            print('(3) node potential')
            for i in nodes.itertuples():
                print(f'   {i.Index}: {int(i[3])}')

            # 4. reduced cost
            print('(4) reduced cost')
            for i in df.itertuples(index=False):
                cij = arcs.loc[(arcs['tail'] == i.tail) & (arcs['head'] == i.head), 'cost'].item()
                cpi = cij - nodes.loc[i.tail, f'{y}_pi'] + nodes.loc[i.head, f'{y}_pi'].item()
                arcs.loc[(arcs['tail'] == i.tail) & (arcs['head'] == i.head), f'{y}_pi'] = cpi
                print(f'   {i.tail},{i.head},{int(cpi)}')

            # 5. find the shortest path and maximum flow
            df = arcs.loc[arcs['capacity'] - arcs['sent'] > 0, ['tail', 'head', f'{y}_pi', 'capacity', 'sent']]
            df.reset_index(drop=True, inplace=True)
            graph = ig.Graph.TupleList(df.itertuples(index=False), directed=True, edge_attrs=[f'{y}_pi'])
            for a in outflow:
                shortest = graph.get_shortest_paths(z[0], to=a[0], weights=graph.es[f'{y}_pi'], output="epath")[0]
                sending = min(df.loc[shortest, 'capacity'])
                sending = min(sending, z[1] - nodes.loc[z[0], 'in'], a[1] - nodes.loc[z[0], 'out'])
                shortest = df.loc[shortest]
                print('(5) find shortest path and maximum flow')
                fprint = f'path: {z[0]}'
                for i in shortest.itertuples(index=False):
                    fprint += f'->{i.head}'
                print(fprint)
                print(f'flow: {sending}\n')

                # 6. sending the flow
                for i in shortest.itertuples(index=False):
                    arcs.loc[(arcs['tail'] == i.tail) & (arcs['head'] == i.head), 'sent'] += sending
                    arcs.loc[(arcs['tail'] == i.head) & (arcs['head'] == i.tail), 'sent'] -= sending
                    if i.tail in inflow_nodes:
                        nodes.loc[i.tail, 'in'] += sending
                    if i.head in outflow_nodes:
                        nodes.loc[i.head, 'out'] += sending
        sent += sending
        x += 1

        # 7. termination conditions
        if sent == total:
            break

    print('algorithm terminates\nResult:\n(1) flow table:')
    total_cost = 0
    for i in arcs.itertuples(index=False):
        if not i.reversed:
            print(f'{i.tail}->{i.head}: flow={i.sent}, cost={i.cost}')
            total_cost += i.sent * i.cost
    print(f'\n(2) summary:\n Flow={sent}\n Total Cost={total_cost}')


def main():
    inputs = pd.read_csv(input_path)
    arcs, nodes, inflow, outflow = build_graph(inputs)
    succesive_shortest_path(arcs, nodes, inflow, outflow)


if __name__ == '__main__':
    main()
