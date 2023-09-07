import click
from demkgtransformer import DemKGTransformer

@click.command()
@click.argument('descriptor', '-d', type=click.Path(exists=True), default='descriptor.yaml')
def main(descriptor):
    transformer = DemKGTransformer(descriptor)
    transformer.transform()
    transformer.save()

if __name__ == '__main__':
    main()